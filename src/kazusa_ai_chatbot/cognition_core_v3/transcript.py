"""Cache-affine transcript assembly for V3 semantic chains and goal branches.

One chain or branch owns one byte-identical static system prompt per family;
question kind, evidence, handles, state projections, stage requests, accepted
projections, and repair facts live only in human-message tails. Same-domain
accepted continuation preserves the complete prior message prefix byte-for-byte.
A route-domain mismatch or context pressure starts a fresh request from a
canonical checkpoint: semantic continuity remains intact while backend KV
continuity ends. Rejected candidates never persist into continuation prefixes;
they appear only inside ephemeral repair requests sent to the provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v3.contracts import CacheDomainIdentity

ROLE_HUMAN = "human"
ROLE_ASSISTANT = "assistant"
_VALID_ROLES: frozenset[str] = frozenset({ROLE_HUMAN, ROLE_ASSISTANT})

TranscriptMessage = tuple[str, str]


class TranscriptContractError(RuntimeError):
    """Fail-closed transcript contract violation."""


@dataclass(frozen=True)
class ChainMessageV1:
    """One immutable primary-chain message row."""

    role: str
    content: str


@dataclass(frozen=True)
class ChainTranscriptV1:
    """Append-only invocation transcript with tail rollback and re-anchor."""

    messages: tuple[ChainMessageV1, ...] = ()
    accepted_products: tuple[Mapping[str, object], ...] = ()
    attempt_ledger: Mapping[str, int] | None = None
    token_ledger: Mapping[str, int] | None = None
    deadline_monotonic: float | None = None
    reanchor_used: bool = False
    pending_interludes: tuple[Mapping[str, object], ...] = ()

    def append_question(self, content: str) -> ChainTranscriptV1:
        """Append one human question, including any pending interludes."""

        if not isinstance(content, str) or not content.strip():
            raise TranscriptContractError(
                "Chain questions must be non-empty strings"
            )
        if self.messages and self.messages[-1].role != ROLE_ASSISTANT:
            raise TranscriptContractError(
                "A new question must follow an accepted assistant answer"
            )
        question = self._prepend_interludes(content)
        return self._with_messages(self.messages + (
            ChainMessageV1(role=ROLE_HUMAN, content=question),
        ))

    def accept_answer(
        self,
        content: str,
        product: Mapping[str, object] | None = None,
    ) -> ChainTranscriptV1:
        """Append one accepted assistant answer and typed product."""

        if not isinstance(content, str) or not content.strip():
            raise TranscriptContractError(
                "Accepted assistant answers must be non-empty strings"
            )
        if not self.messages or self.messages[-1].role != ROLE_HUMAN:
            raise TranscriptContractError(
                "An assistant answer must follow a human question"
            )
        products = self.accepted_products + (product,)
        return self._with_messages(
            self.messages + (
                ChainMessageV1(role=ROLE_ASSISTANT, content=content),
            ),
            products,
        )

    def rollback_tail_answer(self) -> tuple[ChainTranscriptV1, str]:
        """Remove only the current assistant tail and return its question."""

        if len(self.messages) < 2:
            raise TranscriptContractError(
                "A rollback requires an assistant answer after a question"
            )
        if self.messages[-1].role != ROLE_ASSISTANT:
            raise TranscriptContractError(
                "Only the current assistant tail may be rolled back"
            )
        question = self.messages[-2]
        if question.role != ROLE_HUMAN:
            raise TranscriptContractError(
                "The question preceding a rolled-back answer is invalid"
            )
        products = (
            self.accepted_products[:-1]
            if self.accepted_products
            else self.accepted_products
        )
        return (
            self._with_messages(self.messages[:-1], products),
            question.content,
        )

    def append_interlude_to_next_question(
        self,
        interlude: Mapping[str, object],
    ) -> ChainTranscriptV1:
        """Queue one deterministic notice for the next human question."""

        if not isinstance(interlude, Mapping):
            raise TranscriptContractError(
                "Deterministic interludes must be mappings"
            )
        return self._with_pending_interludes(
            self.pending_interludes + (interlude,)
        )

    def reanchor(self, new_system_digest: str) -> ChainTranscriptV1:
        """Replace the message tail with a compact deterministic digest."""

        if self.reanchor_used:
            raise TranscriptContractError("Re-anchor token already consumed")
        if not isinstance(new_system_digest, str) or not new_system_digest.strip():
            raise TranscriptContractError(
                "Re-anchor digests must be non-empty strings"
            )
        return ChainTranscriptV1(
            messages=(),
            accepted_products=self.accepted_products,
            attempt_ledger=self.attempt_ledger,
            token_ledger=self.token_ledger,
            deadline_monotonic=self.deadline_monotonic,
            reanchor_used=True,
            pending_interludes=(),
        ).append_question(new_system_digest)

    def to_messages(self) -> tuple[tuple[str, str], ...]:
        """Expose an immutable role/content projection for callers."""

        return tuple(
            (message.role, message.content)
            for message in self.messages
        )

    def _prepend_interludes(self, content: str) -> str:
        """Prefix queued deterministic notices without a standalone user row."""

        if not self.pending_interludes:
            return content
        interlude_json = ", ".join(
            repr(dict(interlude))
            for interlude in self.pending_interludes
        )
        return f"[interludes: {interlude_json}] {content}"

    def _with_messages(
        self,
        messages: tuple[ChainMessageV1, ...],
        products: tuple[Mapping[str, object], ...] | None = None,
    ) -> ChainTranscriptV1:
        """Build a replacement transcript with queued interludes cleared."""

        return ChainTranscriptV1(
            messages=messages,
            accepted_products=(
                self.accepted_products
                if products is None
                else products
            ),
            attempt_ledger=self.attempt_ledger,
            token_ledger=self.token_ledger,
            deadline_monotonic=self.deadline_monotonic,
            reanchor_used=self.reanchor_used,
            pending_interludes=(),
        )

    def _with_pending_interludes(
        self,
        pending_interludes: tuple[Mapping[str, object], ...],
    ) -> ChainTranscriptV1:
        """Build a replacement transcript with a new interlude queue."""

        return ChainTranscriptV1(
            messages=self.messages,
            accepted_products=self.accepted_products,
            attempt_ledger=self.attempt_ledger,
            token_ledger=self.token_ledger,
            deadline_monotonic=self.deadline_monotonic,
            reanchor_used=self.reanchor_used,
            pending_interludes=pending_interludes,
        )


@dataclass(frozen=True)
class TranscriptState:
    """Persistent cache-affine state for one chain or goal branch.

    ``messages`` holds the post-system history in exact send order; roles are
    restricted to human tails and accepted assistant messages, so a rejected
    candidate has no persistent slot here.
    """

    static_system_prompt: str
    messages: tuple[TranscriptMessage, ...] = ()
    cache_domain_key: str = ""


def start_chain(
    static_system_prompt: str,
    first_human_tail: str,
    identity: CacheDomainIdentity,
) -> TranscriptState:
    """Start one chain or branch transcript from its static system prompt.

    Args:
        static_system_prompt: The byte-identical family-owned static prompt.
        first_human_tail: The stage request plus current facts for the root
            stage; all semantic content lives here, never in the system prefix.
        identity: The validated cache-domain identity of this owner's route.

    Returns:
        The initial transcript state bound to that cache domain key.

    Raises:
        TranscriptContractError: Empty static prompts or empty first tails fail
            fast before any model call is assembled.
    """
    if not static_system_prompt:
        raise TranscriptContractError("Chain transcripts require a non-empty static system prompt")
    if not first_human_tail:
        raise TranscriptContractError("The first human tail must carry the root stage request and facts")

    return TranscriptState(
        static_system_prompt=static_system_prompt,
        messages=((ROLE_HUMAN, first_human_tail),),
        cache_domain_key=identity.domain_key(),
    )


def extend_accepted(
    state: TranscriptState,
    accepted_assistant_text: str,
    next_human_tail: str | None = None,
) -> TranscriptState:
    """Extend a same-domain transcript with an accepted stage candidate.

    The complete prior message prefix is preserved byte-for-byte; the new state
    appends the accepted assistant message and optionally the next stage's human
    tail (request plus compact accepted predecessor projection).

    Args:
        state: The current same-domain transcript state.
        accepted_assistant_text: The validated accepted candidate text.
        next_human_tail: Optional next-stage request with its accepted
            predecessor context; None ends the chain at this stage.

    Returns:
        A new transcript state whose prefix equals the prior messages exactly.
    """
    if not accepted_assistant_text:
        raise TranscriptContractError("Accepted assistant content must be non-empty")

    extension: list[TranscriptMessage] = [(ROLE_ASSISTANT, accepted_assistant_text)]
    if next_human_tail is not None:
        if not next_human_tail:
            raise TranscriptContractError("Next-stage human tails must carry request and facts")
        extension.append((ROLE_HUMAN, next_human_tail))

    return TranscriptState(
        static_system_prompt=state.static_system_prompt,
        messages=state.messages + tuple(extension),
        cache_domain_key=state.cache_domain_key,
    )


def build_repair_request(
    state: TranscriptState,
    invalid_candidate_text: str,
    repair_instruction: str,
) -> Sequence[TranscriptMessage]:
    """Assemble the ephemeral bounded-replacement message sequence.

    The provider sees the persistent prefix plus the latest bounded invalid
    candidate as an assistant turn and the exact contract error with allowed
    values and replacement instruction as a human turn. This sequence is sent
    to the provider only; it never becomes the persistent continuation prefix,
    so rejected candidates are scrubbed before any next-stage extension.

    Args:
        state: The transcript state at the stage request being repaired.
        invalid_candidate_text: The latest bounded structurally invalid candidate.
        repair_instruction: Exact contract error, exact allowed values, and the
            complete-replacement instruction built by the owner's validator.

    Returns:
        The full post-system message sequence for one replacement attempt.

    Raises:
        TranscriptContractError: Empty candidates or empty instructions fail
            fast; a structural repair without its exact error context is
            ambiguous to the owner and is rejected at the boundary.
    """
    if not invalid_candidate_text:
        raise TranscriptContractError("Repair requests require the latest bounded invalid candidate")
    if not repair_instruction:
        raise TranscriptContractError(
            "Repair instructions must carry the exact contract error, allowed values, and replacement instruction"
        )

    return state.messages + ((ROLE_ASSISTANT, invalid_candidate_text), (ROLE_HUMAN, repair_instruction))


def start_fresh_from_checkpoint(
    state: TranscriptState,
    canonical_checkpoint_tail: str,
    identity: CacheDomainIdentity,
) -> TranscriptState:
    """Start a fresh request from a canonical accepted checkpoint.

    Used after a route-domain mismatch or context pressure. The static system
    prompt and chain ownership are unchanged; the new human tail is the
    canonical checkpoint (accepted typed propositions, deltas, semantic
    summaries, and the next owner's prompt-safe projection) plus current facts.
    Backend KV continuity ends at this boundary while semantic continuity stays
    intact.

    Args:
        state: The transcript whose cache domain or budget forced the restart.
        canonical_checkpoint_tail: The assembled checkpoint human tail.
        identity: The validated cache-domain identity of the destination route.

    Returns:
        A fresh two-message transcript (system plus one human tail) bound to the
        destination cache domain key.

    Raises:
        TranscriptContractError: An empty checkpoint tail would silently drop
            accepted state and fails fast.
    """
    if not canonical_checkpoint_tail:
        raise TranscriptContractError("Checkpoint restarts require a non-empty canonical tail")

    return TranscriptState(
        static_system_prompt=state.static_system_prompt,
        messages=((ROLE_HUMAN, canonical_checkpoint_tail),),
        cache_domain_key=identity.domain_key(),
    )


def domain_matches(state: TranscriptState, identity: CacheDomainIdentity) -> bool:
    """Check whether an owner route stays inside the transcript's cache domain.

    Args:
        state: The current transcript state.
        identity: The validated cache-domain identity under check.

    Returns:
        True when both sides resolve to the exact same cache-domain key, so the
        complete prior prefix remains byte-identical and reusable.
    """
    return state.cache_domain_key == identity.domain_key()


def serialized_message_bytes(messages: Sequence[TranscriptMessage]) -> int:
    """Measure one message sequence in UTF-8 bytes for deterministic budgeting.

    Args:
        messages: The post-system transcript sequence under measurement; each
            role tag contributes its own byte length so no boundary can be
            hidden from the count.

    Returns:
        Total UTF-8 byte length of role tags, contents, and one separator per
        message plus a final terminator.
    """
    total = 0
    for role, content in messages:
        if role not in _VALID_ROLES:
            raise TranscriptContractError(f"Unknown transcript role {role!r}")
        total += len(role.encode("utf-8")) + len(content.encode("utf-8")) + 1
    return total


def fits_prompt_budget(
    state: TranscriptState,
    pending_human_tail: str,
    max_total_bytes: int,
) -> bool:
    """Check whether the next extension stays inside a deterministic byte budget.

    Args:
        state: The current transcript state.
        pending_human_tail: The human tail that would be appended by the next
            accepted-stage extension or stage request.
        max_total_bytes: The owner's prompt-budget cap in UTF-8 bytes over the
            static system prompt plus all messages plus the pending tail.

    Returns:
        True when the assembled total stays at or under the cap; False forces a
        canonical checkpoint restart before another model call is issued, so
        context pressure checkpoints before the owner's attempt cap is consumed.
    """
    if max_total_bytes <= 0:
        raise TranscriptContractError("Prompt budgets must be positive byte counts")

    total = (
        len(state.static_system_prompt.encode("utf-8")) + 1
        + serialized_message_bytes(state.messages)
        + len(pending_human_tail.encode("utf-8"))
        + 1
    )
    return total <= max_total_bytes


def to_invoker_messages(
    messages: Sequence[TranscriptMessage],
    static_system_prompt: str | None = None,
) -> list[BaseMessage]:
    """Convert a transcript sequence into invoker-ready LangChain messages.

    Args:
        messages: The post-system message sequence in exact send order.
        static_system_prompt: Optional system prompt to prepend; pass the same
            byte-identical family prompt for every call in one cache domain.

    Returns:
        A fresh list of ``SystemMessage``/``HumanMessage``/``AIMessage`` objects
        with contents copied verbatim, so provider-side prefix bytes match the
        persistent transcript exactly.

    Raises:
        TranscriptContractError: Unknown roles fail fast before any model call.
    """
    result: list[BaseMessage] = []
    if static_system_prompt is not None:
        if not static_system_prompt:
            raise TranscriptContractError("Chain transcripts require a non-empty static system prompt")
        result.append(SystemMessage(content=static_system_prompt))

    for role, content in messages:
        if role == ROLE_HUMAN:
            result.append(HumanMessage(content=content))
        elif role == ROLE_ASSISTANT:
            result.append(AIMessage(content=content))
        else:
            raise TranscriptContractError(f"Unknown transcript role {role!r}")

    return result


__all__ = [
    "ROLE_ASSISTANT",
    "ROLE_HUMAN",
    "ChainMessageV1",
    "ChainTranscriptV1",
    "TranscriptContractError",
    "TranscriptMessage",
    "TranscriptState",
    "build_repair_request",
    "domain_matches",
    "extend_accepted",
    "fits_prompt_budget",
    "serialized_message_bytes",
    "start_chain",
    "start_fresh_from_checkpoint",
    "to_invoker_messages",
]
