#!/usr/bin/env bash
#
# Triggers CI pipeline.

set -eo pipefail

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

die_syntax() {
  echo "Syntax: $0 -p PIPELINE -c COMMIT"
  echo ""
  echo "  -p: Buildkite pipeline"
  echo "  -c: Commit SHA to be built"
  exit 1
}

# Cmd-line opts
while getopts "p:c:" arg; do
  case ${arg} in
    p)
      PIPELINE=${OPTARG}
      ;;
    c)
      COMMIT=${OPTARG}
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
echo "$PIPELINE_CMD" | buildkite-agent pipeline upload
