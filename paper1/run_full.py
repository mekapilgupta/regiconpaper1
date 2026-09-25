#!/usr/bin/env python3
"""
Full Benchmark Experiment Runner (800+ queries per language)
"""

import os
import sys
import json
import argparse
from pathlib import Path

PAPER1_DIR = Path(__file__).resolve().parent
if str(PAPER1_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER1_DIR))

from run_pilot import main as pilot_main
from config import FINAL_SAMPLE_SIZE


def main():
    sys.argv.extend(["--sample_size", str(FINAL_SAMPLE_SIZE), "--output_name", "final_full_experiment"])
    pilot_main()


if __name__ == "__main__":
    main()
