# CSV Normalizer Source-Free Brief

Gate 02 is a source-free task. The coding agent should propose review-only
artifacts for a small Python CLI that normalizes CSV files.

Required behavior:

- read CSV rows from an input path;
- trim whitespace from headers and cell values;
- sort output rows deterministically by all field values;
- support a dry-run mode that prints normalized CSV instead of writing it;
- include focused tests for the normalizer and CLI behavior.
