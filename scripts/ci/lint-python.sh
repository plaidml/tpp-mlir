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

PATTERN="./*.py"
MAXLINELENGTH=79  # suit Flake8-default
LINTER=$(command -v black)

if [ "${LINTER}" ]; then
  FLAKE8=$(command -v flake8)
  FLAKE8_IGNORE="--ignore=E501,F821"
  REPOROOT=$(git_root)
  cd "${REPOROOT}" || exit 1
  echo -n "Checking Python files... "
  for FILE in $(eval "git ls-files ${PATTERN}"); do
    ${LINTER} -l ${MAXLINELENGTH} "${FILE}" -q
    if [ "${FLAKE8}" ]; then  # optional
      # no error raised for Flake8 issues
      WARNING=$(flake8 ${FLAKE8_IGNORE} "${FILE}")
      if [ "${WARNING}" ]; then
        if [ "${WARNINGS}" ]; then
          WARNINGS+=$'\n'"${WARNING}"
        else
          WARNINGS="${WARNING}"
        fi
      fi
    fi
  done

  # any modified file (Git) raises and error
  MODIFIED=$(eval "git ls-files -m ${PATTERN}")
  if [ "${MODIFIED}" ]; then
    echo "ERROR"
    echo
    echo "The following files are modified ($(${LINTER} --version)):"
    echo "${MODIFIED}"
    exit 1
  fi
  # optional warnings
  if [ "${WARNINGS}" ]; then
    echo "WARNING"
    echo
    echo "The following issues were found:"
    echo "${WARNINGS}"
    echo
  else
    echo "OK"
  fi
else
  echo "WARNING: missing Python-linter (${LINTER})."
fi
