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

# If -i is passed, format all files according to type/pattern.
if [ "-i" != "$1" ]; then
  LINTER_FLAGS="-Werror --dry-run"
else
  LINTER_FLAGS="-Werror -i"
fi

COUNT=0; OK=0
echo -n "Linting C/C++ files... "
cd "${REPOROOT}" || exit 1
for FILE in $(eval "git ls-files ${PATTERN}"); do
  if eval "${LINTER} 2>/dev/null ${LINTER_FLAGS} ${FILE}"; then OK=$((OK+1)); fi
  COUNT=$((COUNT+1))
done

if [ "${COUNT}" != "${OK}" ]; then
  echo "ERROR ($((COUNT-OK)) of ${COUNT} files)"
  exit 1
fi

echo "OK (${COUNT} files processed)"
