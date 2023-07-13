#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Reformats Python code belonging to the repository.
# The primary Linter with support for reformatting
# files is Black. Any reformatted code causes
# non-zero exit code (CI failure).
# Additionally, there can be secondary Linters
# used to raise warnings, e.g., Flake8

# include common utils
SCRIPT_DIR=$(realpath "$(dirname "$0")/..")
source "${SCRIPT_DIR}/ci/common.sh"

# maximum line-length to suit Flake8-default
MAXLINELENGTH=79

if [ "$(command -v black)" ]; then
  FLAKE8=$(command -v black)
  FLAKE8_IGNORE="--ignore=E501,F821"
  REPOROOT=$(git_root)
  cd "${REPOROOT}" || exit 1
  echo -n "Checking Python files... "
  for FILE in $(git ls-files ./*.py); do
    black -l ${MAXLINELENGTH} "${FILE}" -q
    if [ "${FLAKE8}" ]; then  # optional
      # no error raised for Flake8 issues
      WARNING=$(flake8 ${FLAKE8_IGNORE} "${FILE}")
      if [ "${WARNING}" ]; then
        if [ "${WARNINGS}" ]; then
          WARNINGS+=$'\n'"${WARNING}"
        else
          WARNINGS=$'\n'"${WARNING}"
        fi
      fi
    fi
  done
  if [ "${WARNINGS}" ]; then
    echo "${WARNINGS}"
    echo "WARNING: discovered Python issues."
  else
    echo "OK"
  fi

  MODIFIED=$(git ls-files -m ./*.py)
  if [ "${MODIFIED}" ]; then
    echo "ERROR: the following files are modified:"
    echo "${MODIFIED}"
    exit 1
  fi
else
  echo "WARNING: missing Python-linter (black)."
fi
