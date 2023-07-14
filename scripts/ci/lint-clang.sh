#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Reformats C/C++ code belonging to the repository.
# The primary Linter with support for reformatting
# files is clang-format. Any reformatted code causes
# non-zero exit code (CI failure).

REPOROOT=$(realpath "$(dirname "$0")/../..")
PATTERN="./*.h ./*.cpp ./*.c"

TPP_CLANGFORMATVER=${TPP_CLANGFORMATVER:-16}
LINTER=$(command -v "clang-format-${TPP_CLANGFORMATVER}")
if [ ! "${LINTER}" ]; then
  LINTER=$(command -v clang-format)
  if ! ${LINTER} --version | grep -q "${TPP_CLANGFORMATVER}\.[0-9\.]\+"; then
    echo "ERROR: clang-format v${TPP_CLANGFORMATVER} is missing!"
    exit 1
  fi
fi

if [ "${LINTER}" ]; then
  COUNT=0

  # If -i is passed, format all files according to type/pattern.
  if [ "-i" != "$1" ]; then
    # list files matching PATTERN and which are part of HEAD's changeset
    LISTFILES="git diff-tree --no-commit-id --name-only HEAD -r"
  else
    LISTFILES="git ls-files"
  fi

  echo -n "Linting C/C++ files... "
  cd "${REPOROOT}" || exit 1
  for FILE in $(eval "${LISTFILES} ${PATTERN}"); do
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
