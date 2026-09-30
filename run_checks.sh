#!/bin/sh
# Run the complete verification without any third-party test runner.
# NumPy is the only runtime dependency.
set -e
cd "$(dirname "$0")"
cd tests && python3 -m unittest discover -v
cd .. && python3 examples/verify_delivery.py
