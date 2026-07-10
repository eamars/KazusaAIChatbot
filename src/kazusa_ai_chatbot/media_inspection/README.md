# Media Inspection ICD

## Purpose

`kazusa_ai_chatbot.media_inspection` is the shared, image-only visual-question
service used below the RAG3 and complex-task resolver families. It returns
evidence and uncertainty; cognition decides stance and dialog owns final text.

## Public Contract

`inspect_media(request)` accepts `media_inspection_request.v1` with a trusted
image MIME type, base64 payload, exact visual question, and optional existing
descriptor. It returns `media_inspection_result.v1` with one of `answered`,
`uncertain`, `unsupported`, `invalid_input`, or `failed`.

The service makes exactly one `VISION_DESCRIPTOR_LLM` multimodal call for a
cache hit. The model receives the image and visual question only. Raw payloads,
cache references, hashes, platform identifiers, database identifiers, and
fetch traces remain outside prompts and user-visible surfaces.

## Session Media

RAG3 stores current and recent images only in a process-local cache scoped by
`(platform, platform_channel_id, global_user_id)`. The fixed configuration is
`MEDIA_SESSION_CACHE_MAX_ITEMS_PER_SCOPE`,
`MEDIA_SESSION_CACHE_MAX_BYTES_PER_SCOPE`,
`MEDIA_SESSION_CACHE_MAX_ITEM_BYTES`, and `MEDIA_SESSION_CACHE_TTL_SECONDS`.
The cache never persists raw media to MongoDB. Cache misses return bounded
missing evidence and make no model call.

## External Media

Complex-task resolver uses its own `media` subagent for public external image
URLs. Deterministic code checks HTTP(S), DNS/IP safety, redirects, timeouts,
MIME, bytes, magic bytes, decode validity, and dimensions before it calls this
shared service. Raw fetched bytes never enter `available_evidence`.
