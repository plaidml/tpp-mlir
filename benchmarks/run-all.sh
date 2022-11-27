#!/bin/bash

source common.sh

echo "SIMPLE COPY:"
pushd simple_copy > /dev/null
./run.sh
popd > /dev/null
echo

echo "MATMUL REF:"
pushd matmul > /dev/null
./run.sh
popd > /dev/null
echo

echo "MLP:"
pushd mlp > /dev/null
./run.sh
popd > /dev/null
echo
