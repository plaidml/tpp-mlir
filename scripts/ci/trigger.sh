#!/usr/bin/env bash
#
# Triggers CI pipeline.

set -eou pipefail

# Create pipeline dynamically
PIPELINE="steps:
  - trigger: $1
    label: $2
    build:
      commit: $2
"

# Upload the new pipeline and add it to the current build
echo "$PIPELINE" | buildkite-agent pipeline upload
