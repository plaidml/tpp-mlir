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
  FLAKE8_IGNORE="--ignore=E402,E501,F821"

  # Check if Flake8 actually works
  if [ "${FLAKE8}" ] && ! ${FLAKE8} 2>/dev/null; then
    unset FLAKE8
  fi

  # If -i is passed, format all files according to type/pattern.
  if [ "-i" != "$1" ]; then
    LINTER_FLAGS="-l 79 -q --check"
  else
    LINTER_FLAGS="-l 79 -q"
  fi

  COUNT=0; OK=0
  echo -n "Linting Python files... "
  cd "${REPOROOT}" || exit 1
  for FILE in $(eval "git ls-files ${PATTERN}"); do
    # Flake8: line-length limit of 79 characters (default)
    if eval "${LINTER} ${LINTER_FLAGS} ${FILE}"; then OK=$((OK+1)); fi
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
    COUNT=$((COUNT+1))
  done

  # mandatory checks
  if [ "${COUNT}" != "${OK}" ]; then
    echo "ERROR ($((COUNT-OK)) of ${COUNT} files)"
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
    echo "OK (${OK} files processed)"
  fi
else  # soft error (exit normally)
  echo "ERROR: missing Python-linter (${LINTER})."
fi
