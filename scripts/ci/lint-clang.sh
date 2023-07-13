#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Reformats C/C++ code belonging to the repository.
# The primary Linter with support for reformatting
# files is clang-format. Any reformatted code causes
# non-zero exit code (CI failure).

REPOROOT=$(realpath "$(dirname "$0")/../..")
PATTERN="./*.h ./*.cpp ./*.c"
VERSION=16
for V in $(seq $((VERSION+10)) -1 ${VERSION}); do
  LINTER=$(command -v "clang-format-${V}")
  if [ "${LINTER}" ]; then break; fi
done
if [ ! "${LINTER}" ]; then
  LINTER=$(command -v clang-format)
fi

if [ "${LINTER}" ]; then
  COUNT=0

  echo -n "Linting C/C++ files... "
  cd "${REPOROOT}" || exit 1
  for FILE in $(eval "git ls-files ${PATTERN}"); do
    if ${LINTER} -i "${FILE}"; then COUNT=$((COUNT+1)); fi
  done

  # any modified file (Git) raises and error
  MODIFIED=$(eval "git ls-files -m ${PATTERN}")
  if [ "${MODIFIED}" ]; then
    echo "ERROR"
    echo
    echo "The following files are modified ($(${LINTER} --version)):"
    echo "${MODIFIED}"
    exit 1
  else
    echo "OK (${COUNT} files)"
  fi
else  # soft error (exit normally)
  echo "ERROR: missing C/C++-linter (${LINTER})."
fi
