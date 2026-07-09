# Release Feed

The package builds a release feed from cached HTTP-style payloads. Two runtime
bugs are seeded:

- cache timeout logic ignores the caller-provided timeout;
- the CLI exposes `--include-drafts` but does not pass it into feed rendering.

The tests are protected for the hard workflow gate.
