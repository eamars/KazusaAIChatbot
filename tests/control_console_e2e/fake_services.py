from __future__ import annotations

import argparse
import signal
import sys
import time


def main() -> int:
    """Run a tiny long-lived service process for lifecycle E2E tests."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    should_run = True

    def request_stop(signum, frame) -> None:
        del signum, frame
        nonlocal should_run
        should_run = False

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(f"{args.name} ready", flush=True)
    while should_run:
        time.sleep(0.1)
    print(f"{args.name} stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
