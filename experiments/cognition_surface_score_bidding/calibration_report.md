# Calibration report

Status: blocked pending trace-backed corpus and independent labels.

The workspace did not contain a trace-backed corpus for either included owner
at execution start. No candidate texts, evaluator outputs, human labels, or
threshold acceptance decisions were synthesized. The two JSONL files therefore
contain explicit zero-count blocked records.

Required completion evidence remains:

- 30 contexts per owner;
- at least two hard-eligible, semantically distinguishable candidates per
  context;
- 20 calibration and 10 disjoint held-out contexts per owner;
- two independent labels per comparison, with adjudication records when
  needed;
- held-out ordering accuracy of at least 80 percent;
- no hard-integrity false acceptance at the selected threshold;
- producer/evaluator call counts and p50/p95 latency for normal and worst-case
  paths; and
- one-at-a-time live verification with inspected raw artifacts.

No production cutover or plan completion can be claimed from this report.
