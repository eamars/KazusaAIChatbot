"""Deterministic tests for V3 cache-affine transcript assembly."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import contracts, transcript

STATIC_PROMPT = "STATIC CHAIN CONTRACT"


def test_tail_rollback_preserves_prefix_and_excludes_rejected_candidate() -> None:
    """Rejected answers are removed while the question prefix stays exact."""

    chain = transcript.ChainTranscriptV1()
    chain = chain.append_question("first stage question")
    prefix = chain.to_messages()
    rejected = chain.accept_answer("rejected candidate", {"step": "A1"})

    repaired, question = rejected.rollback_tail_answer()

    assert question == "first stage question"
    assert repaired.to_messages() == prefix
    joined = "".join(content for _, content in repaired.to_messages())
    assert "rejected candidate" not in joined
    assert repaired.accepted_products == ()


def _identity(url: str = "https://backend.test/v1", credential: str = "cred-a") -> contracts.CacheDomainIdentity:
    return contracts.validate_cache_domain_identity(
        contracts.CacheDomainIdentity(
            normalized_backend_url=url,
            credential_identity_hash=contracts.hash_credential_identity(credential),
            backend_kind="openai_compatible",
            model="local-model",
            template_strategy="chat",
            static_system_prompt_hash=contracts.hash_static_prompt(STATIC_PROMPT),
        )
    )


def test_accepted_same_domain_stage_extends_exact_prefix():
    identity = _identity()
    state = transcript.start_chain(STATIC_PROMPT, "stage 1 current facts", identity)

    extended = transcript.extend_accepted(state, "stage 1 accepted candidate", "stage 2 request plus compact accepted projection")

    assert extended.messages[: len(state.messages)] == state.messages
    assert extended.static_system_prompt == state.static_system_prompt
    assert extended.cache_domain_key == identity.domain_key()
    assert extended.messages[len(state.messages):] == (
        ("assistant", "stage 1 accepted candidate"),
        ("human", "stage 2 request plus compact accepted projection"),
    )

    invoker_messages = transcript.to_invoker_messages(extended.messages, STATIC_PROMPT)
    contents = [message.content for message in invoker_messages]
    assert contents == [
        STATIC_PROMPT,
        "stage 1 current facts",
        "stage 1 accepted candidate",
        "stage 2 request plus compact accepted projection",
    ]

    with pytest.raises(transcript.TranscriptContractError, match="non-empty static system prompt"):
        transcript.start_chain("", "tail", identity)


def test_rejected_candidate_is_scrubbed_before_next_stage():
    identity = _identity()
    state = transcript.start_chain(STATIC_PROMPT, "stage request plus facts", identity)

    repair_sequence = transcript.build_repair_request(state, "latest bounded invalid candidate", "exact contract error plus allowed values plus replacement instruction")
    assert repair_sequence[: len(state.messages)] == state.messages
    assert repair_sequence[len(state.messages):] == (
        ("assistant", "latest bounded invalid candidate"),
        ("human", "exact contract error plus allowed values plus replacement instruction"),
    )

    accepted = transcript.extend_accepted(
        state,
        "complete accepted replacement",
        "next stage request",
    )
    joined = "".join(content for _, content in accepted.messages)
    assert "latest bounded invalid candidate" not in joined
    assert "exact contract error plus allowed values" not in joined
    assert accepted.messages[: len(state.messages)] == state.messages

    with pytest.raises(transcript.TranscriptContractError, match="invalid candidate"):
        transcript.build_repair_request(state, "", "instruction")

    with pytest.raises(transcript.TranscriptContractError, match="contract error"):
        transcript.build_repair_request(state, "candidate", "")


def test_cache_domain_change_forces_canonical_checkpoint():
    identity_a = _identity(credential="cred-a")
    identity_b = _identity(url="https://other-backend.test/v1", credential="cred-b")

    state = transcript.start_chain(STATIC_PROMPT, "stage 1 current facts", identity_a)
    extended = transcript.extend_accepted(state, "stage 1 accepted candidate", None)

    assert not transcript.domain_matches(extended, identity_b)

    checkpoint_tail = (
        "canonical accepted checkpoint: stage 1 accepted propositions and bounded summary"
        " plus next stage current facts"
    )
    fresh = transcript.start_fresh_from_checkpoint(extended, checkpoint_tail, identity_b)

    assert fresh.static_system_prompt == STATIC_PROMPT
    assert fresh.messages == (("human", checkpoint_tail),)
    assert fresh.cache_domain_key == identity_b.domain_key()
    joined = "".join(content for _, content in fresh.messages)
    assert "stage 1 accepted propositions" in joined
    assert "stage 1 current facts" not in joined


def test_context_pressure_checkpoints_before_owner_cap():
    identity = _identity()
    first_tail = "root stage request plus full current evidence and state projection"
    state = transcript.start_chain(STATIC_PROMPT, first_tail, identity)

    small_tail = "next stage request within budget"
    max_bytes = (
        len(STATIC_PROMPT.encode("utf-8")) + 1
        + transcript.serialized_message_bytes(state.messages)
        + len(small_tail.encode("utf-8"))
        + 2
    )

    assert transcript.fits_prompt_budget(state, small_tail, max_bytes)
    oversized_tail = small_tail + "padded-beyond-budget" * 40
    assert not transcript.fits_prompt_budget(state, oversized_tail, max_bytes)

    compact_checkpoint = "accepted cp plus facts"
    assert len(compact_checkpoint.encode("utf-8")) < len(first_tail.encode("utf-8"))
    fresh = transcript.start_fresh_from_checkpoint(
        state,
        compact_checkpoint,
        identity,
    )
    assert transcript.fits_prompt_budget(fresh, small_tail, max_bytes)
