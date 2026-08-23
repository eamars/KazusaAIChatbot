"""Public direct-call runtime composition and resolve lifecycle."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from types import MappingProxyType

from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverLimitsV1,
    AgenticResolverRequestV1,
    AgenticResolverResultV1,
    AgenticResolverSubagentTaskV1,
    validated_request,
)
from agentic_resolver.json_protocol import (
    skill_catalog_message,
    subagent_task_message,
    system_policy_message,
    task_message,
)
from agentic_resolver.loop import AgentLoop
from agentic_resolver.model import (
    AgenticModelCapabilitiesV1,
    AgenticModelClient,
)
from agentic_resolver.session import ResolverSession
from agentic_resolver.skills import SkillCatalog
from agentic_resolver.subagents import SubagentRunner
from agentic_resolver.tools import ToolRegistry


class AgenticResolverRuntime:
    """Construct isolated root and child sessions over fixed dependencies."""

    def __init__(
        self,
        *,
        model: AgenticModelClient,
        tools: ToolRegistry,
        skills: SkillCatalog,
        limits: AgenticResolverLimitsV1 | None = None,
        permission_scope: Mapping[str, object] | None = None,
    ) -> None:
        capabilities = model.capabilities
        if not isinstance(capabilities, AgenticModelCapabilitiesV1):
            raise AgenticResolverContractError(
                "model must expose AgenticModelCapabilitiesV1",
                code="unsupported_model_capability",
            )
        if not isinstance(tools, ToolRegistry):
            raise AgenticResolverContractError(
                "tools must be a frozen ToolRegistry"
            )
        if not isinstance(skills, SkillCatalog):
            raise AgenticResolverContractError(
                "skills must be an immutable SkillCatalog"
            )
        self._model = model
        self._ordinary_tools = tools
        self._root_tools = tools.with_core_tools(include_subagent=True)
        self._child_tools = tools.with_core_tools(include_subagent=False)
        self._skills = skills
        self._limits = limits or AgenticResolverLimitsV1()
        if len(skills.definitions) > self._limits.max_skills:
            raise AgenticResolverContractError(
                "skill catalog exceeds runtime limits"
            )
        self._permission_scope = MappingProxyType(
            dict(permission_scope or {})
        )
        self._subagent_runner = SubagentRunner(self)
        self._sessions: dict[str, ResolverSession] = {}

    @property
    def limits(self) -> AgenticResolverLimitsV1:
        """Return the fixed caller-lowered runtime limits."""

        return self._limits

    @property
    def permission_scope(self) -> Mapping[str, object]:
        """Return the trusted permission scope inherited by each child."""

        return self._permission_scope

    async def resolve(
        self,
        request: AgenticResolverRequestV1,
    ) -> AgenticResolverResultV1:
        """Resolve one prompt-safe task without service or workflow startup."""

        normalized_request = validated_request(request)
        session_id = f"resolver-{uuid.uuid4().hex}"
        deadline = time.monotonic() + self._limits.session_timeout_seconds
        result = await self._resolve_session(
            request=normalized_request,
            session_id=session_id,
            depth=0,
            parent_session_id=None,
            task_content=task_message(normalized_request),
            tools=self._root_tools,
            deadline=deadline,
        )
        return result

    async def _resolve_child(
        self,
        *,
        request: AgenticResolverRequestV1,
        task: AgenticResolverSubagentTaskV1,
        subagent_id: str,
        parent_session_id: str,
        deadline: float,
    ) -> AgenticResolverResultV1:
        """Create one isolated depth-one session through the same runtime path."""

        result = await self._resolve_session(
            request=request,
            session_id=subagent_id,
            depth=1,
            parent_session_id=parent_session_id,
            task_content=subagent_task_message(task),
            tools=self._child_tools,
            deadline=deadline,
        )
        return result

    async def _resolve_session(
        self,
        *,
        request: AgenticResolverRequestV1,
        session_id: str,
        depth: int,
        parent_session_id: str | None,
        task_content: str,
        tools: ToolRegistry,
        deadline: float,
    ) -> AgenticResolverResultV1:
        """Run one root or child through the common serialized AgentLoop."""

        catalog_content = skill_catalog_message(
            catalog_digest=self._skills.catalog_digest,
            skills=self._skills.summaries(),
        )
        session = ResolverSession(
            session_id=session_id,
            depth=depth,
            parent_session_id=parent_session_id,
            policy_content=system_policy_message(),
            catalog_content=catalog_content,
            task_content=task_content,
        )
        self._sessions[session_id] = session
        loop = AgentLoop(
            model=self._model,
            registry=tools,
            skills=self._skills,
            limits=self._limits,
            permission_scope=self._permission_scope,
            subagent_runner=self._subagent_runner,
        )
        result = await loop.run(
            session=session,
            request=request,
            deadline=deadline,
        )
        return result
