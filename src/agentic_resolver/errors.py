"""Closed standalone resolver error taxonomy."""

from __future__ import annotations

from enum import Enum


class RuntimeFaultCode(str, Enum):
    """Machine-readable runtime faults returned across the RPC boundary."""

    ACTION_CONTRACT_EXHAUSTED = "RESOLVER_ACTION_CONTRACT_EXHAUSTED"
    OPERATION_OUTCOME_UNCERTAIN = "OPERATION_OUTCOME_UNCERTAIN"
    OPERATION_ID_REUSE_MISMATCH = "OPERATION_ID_REUSE_MISMATCH"
    STALE_ACTIVATION_OR_LEASE = "STALE_ACTIVATION_OR_LEASE"
    DUPLICATE_ACTIVATION = "DUPLICATE_ACTIVATION"
    RPC_AUTHENTICATION_FAILED = "RPC_AUTHENTICATION_FAILED"
    RPC_CONTRACT_ERROR = "RPC_CONTRACT_ERROR"
    RPC_TRANSPORT_ERROR = "RPC_TRANSPORT_ERROR"
    RESOLUTION_PERSISTENCE_ERROR = "RESOLUTION_PERSISTENCE_ERROR"
    TERMINAL_RECEIPT_MISSING = "TERMINAL_RECEIPT_MISSING"
    TERMINAL_RECEIPT_INVALID = "TERMINAL_RECEIPT_INVALID"


class InteractionFaultCode(str, Enum):
    """Machine-readable Brain interaction faults."""

    AUTHENTICATION_FAILED = "BRAIN_INTERACTION_AUTHENTICATION_FAILED"
    REPLAY = "BRAIN_INTERACTION_REPLAY"
    EXPIRED = "BRAIN_INTERACTION_EXPIRED"
    UNAVAILABLE = "BRAIN_INTERACTION_UNAVAILABLE"
    IDENTITY_INVALID = "BRAIN_INTERACTION_IDENTITY_INVALID"


class AgenticResolverError(RuntimeError):
    """Base class for typed resolver integration errors."""

    code = "AGENTIC_RESOLVER_ERROR"


class ResolverContractError(AgenticResolverError, ValueError):
    """Raised when a strict Kazusa DTO violates its declared schema."""

    code = RuntimeFaultCode.RPC_CONTRACT_ERROR.value


class RpcContractError(ResolverContractError):
    """Raised for malformed JSON-RPC request or response frames."""


class RpcAuthenticationError(AgenticResolverError, PermissionError):
    """Raised when the RPC bearer credential is absent or invalid."""

    code = RuntimeFaultCode.RPC_AUTHENTICATION_FAILED.value


class RpcTransportError(AgenticResolverError, ConnectionError):
    """Raised for a bounded loopback transport failure."""

    code = RuntimeFaultCode.RPC_TRANSPORT_ERROR.value


class OperationIdReuseMismatchError(AgenticResolverError):
    """Raised when one semantic operation id is reused with another digest."""

    code = RuntimeFaultCode.OPERATION_ID_REUSE_MISMATCH.value


class OperationOutcomeUncertainError(AgenticResolverError):
    """Raised when inspection cannot determine an ambiguous outcome."""

    code = RuntimeFaultCode.OPERATION_OUTCOME_UNCERTAIN.value


class StaleActivationOrLeaseError(AgenticResolverError):
    """Raised before a stale live-control request can mutate sidecar state."""

    code = RuntimeFaultCode.STALE_ACTIVATION_OR_LEASE.value


class DuplicateActivationError(AgenticResolverError):
    """Raised when a segment already has a non-expired activation."""

    code = RuntimeFaultCode.DUPLICATE_ACTIVATION.value


class ResolutionPersistenceError(AgenticResolverError):
    """Raised for a failed or inconsistent lifecycle metadata mutation."""

    code = RuntimeFaultCode.RESOLUTION_PERSISTENCE_ERROR.value
