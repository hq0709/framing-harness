#!/usr/bin/env bash
# What a compile check does not catch.
#
# Every script here compiled cleanly while four of them were missing an import: a NameError is a runtime
# error, and these scripts run their argument parser at module level, so even --help exits before the first
# undefined name is reached. The only thing that found it was a static scan, after the repository had
# already been pushed. So the scan is a command now.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== undefined names, unused imports"
python -m pyflakes src experiments analysis
echo "== every script parses its arguments"
for f in experiments/*.py experiments/exploratory/*.py; do
  python "$f" --help > /dev/null 2>&1 || echo "  no --help: $f"
done
echo "== the tables rebuild from the runs in results/"
python analysis/stability.py | tail -3
echo "OK"
