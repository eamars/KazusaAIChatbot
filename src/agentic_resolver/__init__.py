"""Canonical public facade for the standalone DSH resolver."""

from agentic_resolver.contracts import (
    DSHResolutionExhaustV1,
    DSHResolutionIntakeV1,
    DSHResolutionModelInputV1,
    DSHResolutionRuntimeV1,
    EvidenceReferenceV1,
    ResolutionThreadRecordV1,
    SubmitResolutionV1,
)
from agentic_resolver.controller import ResolutionController
from agentic_resolver.errors import (
    AgenticResolverError,
    DuplicateActivationError,
    OperationIdReuseMismatchError,
    OperationOutcomeUncertainError,
    ResolverContractError,
    RpcAuthenticationError,
    RpcContractError,
    RpcTransportError,
    RuntimeFaultCode,
    StaleActivationOrLeaseError,
)
from agentic_resolver.runtime import AgenticResolverRuntime

__all__ = [
    "AgenticResolverError",
    "AgenticResolverRuntime",
    "DSHResolutionExhaustV1",
    "DSHResolutionIntakeV1",
    "DSHResolutionModelInputV1",
    "DSHResolutionRuntimeV1",
    "DuplicateActivationError",
    "EvidenceReferenceV1",
    "OperationIdReuseMismatchError",
    "OperationOutcomeUncertainError",
    "ResolutionController",
    "ResolutionThreadRecordV1",
    "ResolverContractError",
    "RpcAuthenticationError",
    "RpcContractError",
    "RpcTransportError",
    "RuntimeFaultCode",
    "StaleActivationOrLeaseError",
    "SubmitResolutionV1",
]
