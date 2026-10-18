#!/usr/bin/env python3
"""Make a test corpus that behaves like the graded one.

    python3 scripts/prepare_corpus.py --src mc3-starter-kit/mc3-corpus --dest /tmp/corpus

Copies the corpus, recreates the EMPTY directory a zip cannot carry (``archive/``) and
makes ``vendor/internal_audit.txt`` unreadable (chmod 000). As root, chmod 000 does not
stop reading; the harness drops DAC_OVERRIDE, so test with
``docker run --cap-drop DAC_OVERRIDE ...`` (scripts/check_submission.sh does).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    if args.dest.exists():
        # Files may be mode 000 from an earlier run; make them deletable first.
        for path in args.dest.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o644)
        shutil.rmtree(args.dest)
    shutil.copytree(args.src, args.dest)
    (args.dest / "archive").mkdir(exist_ok=True)
    audit = args.dest / "vendor" / "internal_audit.txt"
    if audit.exists():
        audit.chmod(0o000)
    print(f"corpus ready at {args.dest} (archive/ empty, {audit.name} chmod 000)")
    if os.geteuid() == 0:
        print(
            "NOTE: you are root, so chmod 000 will not stop reading. Use --cap-drop DAC_OVERRIDE."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
