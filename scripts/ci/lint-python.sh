#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Reformats Python code belonging to the repository.
# The primary Linter with support for reformatting
# files is Black. Any reformatted code causes
# non-zero exit code (CI failure).
# Additionally, there can be secondary Linters
# used to raise warnings, e.g., Flake8

REPOROOT=$(realpath "$(dirname "$0")/../..")
PATTERN="./*.py"
LINTER=$(command -v black)

if [ "${LINTER}" ]; then
  FLAKE8=$(command -v flake8)
  FLAKE8_IGNORE="--ignore=E501,F821"

  echo -n "Linting Python files... "
  cd "${REPOROOT}" || exit 1
  for FILE in $(eval "git ls-files ${PATTERN}"); do
    # Flake8: line-length limit of 79 characters (default)
    ${LINTER} -q -l 79 "${FILE}"
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
