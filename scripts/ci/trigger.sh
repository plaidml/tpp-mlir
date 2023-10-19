#!/usr/bin/env bash
#
# Triggers CI pipeline.

set -eo pipefail

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

die_syntax() {
  echo "Syntax: $0 -p PIPELINE -c COMMIT [-l lock-name]"
  echo ""
  echo "  -p: Buildkite pipeline"
  echo "  -c: Git commit SHA to be built"
  echo "  -l: Optional, uses buildkite lock"
  exit 1
}

# Cmd-line opts
while getopts "p:c:l:" arg; do
  case ${arg} in
    p)
      PIPELINE=${OPTARG}
      ;;
    c)
      COMMIT=${OPTARG}
      ;;
    l)
      LOCK=${OPTARG}
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

# Mandatory arguments
if [ ! "${PIPELINE}" ] || [ ! "${COMMIT}" ]; then
  die_syntax
fi

# Create pipeline dynamically
PIPELINE_CMD="steps:
  - trigger: ${PIPELINE}
    label: ${COMMIT}
    build:
      commit: ${COMMIT}
"

# Upload the new pipeline and add it to the current build.
# Simply run the job if no lock is defined.
if [ ! "${LOCK}" ]; then
  echo "$PIPELINE_CMD" | buildkite-agent pipeline upload
else
  BUILD_LOCK=$(buildkite-agent lock do ${LOCK})
  if [ ${BUILD_LOCK} == 'do' ]; then
    echo "$PIPELINE_CMD" | buildkite-agent pipeline upload
    buildkite-agent lock done ${LOCK}
  fi
fi
