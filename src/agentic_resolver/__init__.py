"""Canonical public facade for the standalone DSH resolver."""

from agentic_resolver.contracts import (
    DSHResolutionExhaustV2,
    DSHResolutionIntakeV2,
    DSHResolutionModelInputV2,
    DSHResolutionRuntimeV2,
    EvidenceReceiptV2,
    ResolutionThreadRecordV2,
    SubmitResolutionV2,
)
from agentic_resolver.controller import ResolutionController
from agentic_resolver.errors import (
    AgenticResolverError,
    DuplicateActivationError,
    InteractionFaultCode,
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
    "DSHResolutionExhaustV2",
    "DSHResolutionIntakeV2",
    "DSHResolutionModelInputV2",
    "DSHResolutionRuntimeV2",
    "DuplicateActivationError",
    "EvidenceReceiptV2",
    "InteractionFaultCode",
    "OperationIdReuseMismatchError",
    "OperationOutcomeUncertainError",
    "ResolutionController",
    "ResolutionThreadRecordV2",
    "ResolverContractError",
    "RpcAuthenticationError",
    "RpcContractError",
    "RpcTransportError",
    "RuntimeFaultCode",
    "StaleActivationOrLeaseError",
    "SubmitResolutionV2",
]
