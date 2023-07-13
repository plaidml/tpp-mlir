#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Reformats C/C++ code belonging to the repository.
# The primary Linter with support for reformatting
# files is clang-format. Any reformatted code causes
# non-zero exit code (CI failure).
# Additionally, there can be secondary Linters
# used to raise warnings.

# include common utils
SCRIPT_DIR=$(realpath "$(dirname "$0")/..")
source "${SCRIPT_DIR}/ci/common.sh"

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
  #OTHER=$(command -v other)
  REPOROOT=$(git_root)
  cd "${REPOROOT}" || exit 1
  echo -n "Linting C/C++ files... "
  for FILE in $(eval "git ls-files ${PATTERN}"); do
    ${LINTER} -i "${FILE}"
    if [ "${OTHER}" ]; then  # optional
      # no error raised for other issues
      WARNING=$(other "${FILE}")
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
  echo "WARNING: missing C/C++-linter (${LINTER})."
fi
