#!/usr/bin/env python3
"""入口: python3 scripts/analyze_record.py --record X --checkpoint Y --out Z"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from jiangpu.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
