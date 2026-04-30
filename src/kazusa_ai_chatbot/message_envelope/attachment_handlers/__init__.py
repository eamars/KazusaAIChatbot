"""Attachment handler implementations for message-envelope normalization.

This package is the extension slot for concrete `AttachmentHandler`
implementations. Inputs are adapter-provided attachment objects; outputs are
storable `AttachmentRef` payloads. Consumers outside `message_envelope` depend
on the public Protocol and registry, not concrete modules in this package.
"""
