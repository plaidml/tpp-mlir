#!/usr/bin/env bash
#
# Check and install all dependencies

# Program deps
check_program() {
  PROG=$1
  if not which $PROG > /dev/null; then
    echo "Required program '$PROG' not found"
    exit 1
  fi
}
check_program pip
check_program cmake
check_program ninja

# Python deps
pip install --upgrade --user lit
check_program lit

echo "All dependencies satisfied"
