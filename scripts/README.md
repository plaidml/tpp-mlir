# Scripts

These scripts help build and maintain the project.

The subdirectories are:

# CI

Generic CI scripts that check for dependencies, prepare environments (Conda, Virtualenv, etc).
They can be used by multiple CI environments to do generic setup, not specific setup.
They can also be used by developers on their machines.

Scripts to CMake and build the project, run benchmarks etc.
These should be generic to all environments, including developers' own machines.
Given the appropriate dependencies are installed, these should work everywhere.

# Buildkite

Scripts executed exclusively by our buildkite CI.
There should be one script per rule, used by all different builds.

There are two types of scripts here:
 * YAML: Pipeline descrptions that the buildkite-agent will upload.
 * BASH: Scripts that the YAML rules will call.

There should be no code in the YAML files.
The bash scripts here should call the other generic scripts elsewhere.
