// RUN: tpp-opt %s -pack-unpack-optimization -split-input-file | FileCheck %s

func.func @generalize_pack_unpack(%arg0: tensor<12x2x56x56x32xf32>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<256x1024xbf16>)
                          -> (tensor<256x1024x2xbf16>, tensor<12x56x56x64xf32>) {
  %packOut = tensor.empty() : tensor<256x1024x2xbf16>
  %0 = tensor.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %packOut : tensor<512x1024xbf16> -> tensor<256x1024x2xbf16>
  %unpackOut = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %unpackOut : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  return %0, %1 : tensor<256x1024x2xbf16>, tensor<12x56x56x64xf32>
}

// CHECK-LABEL: func.func @generalize_pack_unpack(
// CHECK-SAME: %[[ARG0:.+]]: tensor<12x2x56x56x32xf32>, %[[ARG1:.+]]: tensor<512x1024xbf16>, %[[ARG2:.+]]: tensor<256x1024xbf16>
// CHECK-NOT: tensor.pack
// CHECK-DAG:  %[[c56:.+]] = arith.constant 56 : index
// CHECK-DAG:  %[[c12:.+]] = arith.constant 12 : index
// CHECK-DAG:  %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:  %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:  %[[c256:.+]] = arith.constant 256 : index
// CHECK-DAG:  %[[c1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:  %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG:  %[[c32:.+]] = arith.constant 32 : index
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<256x1024x2xbf16>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c256]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<256x1024x2xbf16>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c1024]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<256x1024x2xbf16>) {
// CHECK:     %[[MUL:.+]] = arith.muli %[[ARG3]], %[[c2]] : index
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[MUL]], %[[ARG5]]] [2, 1] [1, 1] : tensor<512x1024xbf16> to tensor<2xbf16>
// CHECK:     tensor.insert_slice %[[EXTRACT]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0] [1, 1, 2] [1, 1, 1] : tensor<2xbf16> into tensor<256x1024x2xbf16>
// CHECK-NOT: tensor.unpack
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c12]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c56]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c56]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:       scf.for %[[ARG9:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:         %[[MUL0:.+]] = arith.muli %[[ARG9]], %[[c32]] : index
// CHECK:         %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARG9]], %[[ARG5]], %[[ARG7]], 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : tensor<12x2x56x56x32xf32> to tensor<32xf32>
// CHECK:   %[[INSERTED:.+]] = tensor.insert_slice %[[EXTRACT]] into %[[ARG10]][%[[ARG3]], %[[ARG5]], %[[ARG7]], %[[MUL0]]] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<32xf32> into tensor<12x56x56x64xf32>

// -----

func.func @unpack1(%in: tensor<2x2x2x2xf32>, %out: tensor<4x4xf32>) ->  tensor<4x4xf32> {
  %1 = tensor.unpack %in inner_dims_pos = [0, 1] inner_tiles = [2,2] into %out : tensor<2x2x2x2xf32> -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK: func.func @unpack1(%[[ARG0:.+]]: tensor<2x2x2x2xf32>, %[[ARG1:.+]]: tensor<4x4xf32>) ->  tensor<4x4xf32> {
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK: scf.for %[[ARG2:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG3:.+]] = %[[ARG1]]) -> (tensor<4x4xf32>) {
// CHECK:  scf.for %[[ARG4:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG5:.+]] = %[[ARG3]]) -> (tensor<4x4xf32>) {
// CHECK:    %[[MUL0:.+]] = arith.muli %[[ARG2]], %[[c2]] : index
// CHECK:    %[[MUL1:.+]] = arith.muli %[[ARG4]], %[[c2]] : index
// CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], %[[ARG4]], 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<2x2x2x2xf32> to tensor<2x2xf32>
// CHECK:    tensor.insert_slice %[[EXTRACT]] into %[[ARG5]][%[[MUL0]], %[[MUL1]]] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x4xf32>

// -----

func.func @unpack2(%0: tensor<1x2x2x2x2xf32>, %1: tensor<1x2x2x4xf32>)-> tensor<1x2x2x4xf32>{
 %2 = tensor.unpack %0  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %1 : tensor<1x2x2x2x2xf32> -> tensor<1x2x2x4xf32>
  return %2: tensor<1x2x2x4xf32>
}

// CHECK: func.func @unpack2(%[[ARG0:.+]]: tensor<1x2x2x2x2xf32>, %[[ARG1:.+]]: tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32> {
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK: scf.for %[[ARG2:.+]] = %[[c0]] to %[[c1]] step %[[c1]] iter_args(%[[ARG3:.+]] = %[[ARG1]]) -> (tensor<1x2x2x4xf32>) {
// CHECK: scf.for %[[ARG4:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG5:.+]] = %[[ARG3]]) -> (tensor<1x2x2x4xf32>) {
// CHECK:   scf.for %[[ARG6:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG7:.+]] = %[[ARG5]]) -> (tensor<1x2x2x4xf32>) {
// CHECK:     scf.for %[[ARG8:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG9:.+]] = %[[ARG7]]) -> (tensor<1x2x2x4xf32>) {
// CHECK:       %[[MUL:.+]] = arith.muli %[[ARG8]], %[[c2]] : index
// CHECK:       %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], %[[ARG8]], %[[ARG4]], %[[ARG6]], 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : tensor<1x2x2x2x2xf32> to tensor<2xf32>
// CHECK:       tensor.insert_slice %[[EXTRACT]] into %[[ARG9]][%[[ARG2]], %[[ARG4]], %[[ARG6]], %[[MUL]]] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<2xf32> into tensor<1x2x2x4xf32>

// -----

func.func @pack1(%in: tensor<4x4xf32>, %out: tensor<2x2x2x2xf32>) ->  tensor<2x2x2x2xf32> {
  %1 = tensor.pack %in inner_dims_pos = [0, 1] inner_tiles = [2,2] into %out : tensor<4x4xf32> -> tensor<2x2x2x2xf32>
  return %1 : tensor<2x2x2x2xf32>
}

// CHECK: func.func @pack1(%[[ARG0:.+]]: tensor<4x4xf32>, %[[ARG1:.+]]: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK: scf.for %[[ARG2:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG3:.+]] = %[[ARG1]]) -> (tensor<2x2x2x2xf32>) {
// CHECK:   scf.for %[[ARG4:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG5:.+]] = %[[ARG3]]) -> (tensor<2x2x2x2xf32>) {
// CHECK:     %[[MUL0:.+]] = arith.muli %[[ARG2]], %[[c2]] : index
// CHECK:     %[[MUL1:.+]] = arith.muli %[[ARG4]], %[[c2]] : index
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[MUL0]], %[[MUL1]]] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK:     tensor.insert_slice %[[EXTRACT]] into %[[ARG5]][%[[ARG2]], %[[ARG4]], 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2x2xf32>

// -----

func.func @pack2(%0: tensor<1x2x2x4xf32>, %1:  tensor<1x2x2x2x2xf32>)-> tensor<1x2x2x2x2xf32>{
 %2 = tensor.pack %0  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %1 : tensor<1x2x2x4xf32> -> tensor<1x2x2x2x2xf32>
  return %2: tensor<1x2x2x2x2xf32>
}

// CHECK: func.func @pack2(%[[ARG0:.+]]: tensor<1x2x2x4xf32>, %[[ARG1:.+]]: tensor<1x2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32> {
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK: scf.for %[[ARG2:.+]] = %[[c0]] to %[[c1]] step %[[c1]] iter_args(%[[ARG3:.+]] = %[[ARG1]]) -> (tensor<1x2x2x2x2xf32>) {
// CHECK: scf.for %[[ARG4:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG5:.+]] = %[[ARG3]]) -> (tensor<1x2x2x2x2xf32>) {
// CHECK:   scf.for %[[ARG6:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG7:.+]] = %[[ARG5]]) -> (tensor<1x2x2x2x2xf32>) {
// CHECK:     scf.for %[[ARG8:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG9:.+]] = %[[ARG7]]) -> (tensor<1x2x2x2x2xf32>) {
// CHECK:       %[[MUL0:.+]] = arith.muli %[[ARG8]], %[[c2]] : index
// CHECK:       %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], %[[ARG4]], %[[ARG6]], %[[MUL0]]] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<2xf32>
// CHECK        tensor.insert_slice %[[EXTRACT]] into %[[ARG7]][0, %[[ARG6]], %[[ARG2]], %[[ARG4]], 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : tensor<2xf32> into tensor<1x2x2x2x2xf32>
