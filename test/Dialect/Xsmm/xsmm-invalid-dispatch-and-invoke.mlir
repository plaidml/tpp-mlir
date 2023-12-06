// RUN: tpp-opt %s -verify-xsmm-calls -verify-diagnostics -split-input-file

func.func @gemm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (none) data_type = f32
  // expected-error@+1 {{invalid dispatch operation}}
  xsmm.gemm(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (vnni_a) data_type = bf16
  // expected-error@+1 {{inconsistent data types}}
  xsmm.gemm(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>) {
  %0 = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (vnni_a) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand A or invalid VNNI_A flags}}
  xsmm.gemm(data_type = bf16, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> ()
  return
}

// -----

func.func @gemm(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>) {
  %0 = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (vnni_b) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand B or invalid VNNI_B flags}}
  xsmm.gemm(data_type = bf16, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> ()
  return
}

// -----

func.func @gemm(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>) {
  %0 = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (vnni_c) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand C or invalid VNNI_C flags}}
  xsmm.gemm(data_type = bf16, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (none) data_type = f32
  %1 = arith.constant 1 : i64
  // expected-error@+1 {{invalid dispatch operation}}
  xsmm.brgemm(data_type = f32, %0, %arg0, %arg0, %arg1, %1) :
    (i64, memref<1x3x3xf32>, memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (none) data_type = bf16
  // expected-error@+1 {{inconsistent data types}}
  xsmm.brgemm(data_type = f32, %0, %arg0, %arg0, %arg1, %0) :
    (i64, memref<1x3x3xf32>, memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xbf16>, %arg1: memref<3x3xbf16>, %batch : i64) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (vnni_a) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand A or invalid VNNI_A flags}}
  xsmm.brgemm(data_type = bf16, %0, %arg0, %arg0, %arg1, %batch) :
    (i64, memref<1x3x3xbf16>, memref<1x3x3xbf16>, memref<3x3xbf16>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xbf16>, %arg1: memref<3x3xbf16>, %batch : i64) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (vnni_b) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand B or invalid VNNI_B flags}}
  xsmm.brgemm(data_type = bf16, %0, %arg0, %arg0, %arg1, %batch) :
    (i64, memref<1x3x3xbf16>, memref<1x3x3xbf16>, memref<3x3xbf16>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xbf16>, %arg1: memref<3x3xbf16>, %batch : i64) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (vnni_c) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand C or invalid VNNI_C flags}}
  xsmm.brgemm(data_type = bf16, %0, %arg0, %arg0, %arg1, %batch) :
    (i64, memref<1x3x3xbf16>, memref<1x3x3xbf16>, memref<3x3xbf16>, i64) -> ()
  return
}

// -----

func.func @fused_brgemm(%arg0: memref<1x3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3xf32>) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (none) data_type = f32
  // expected-error@+1 {{invalid dispatch operation}}
  xsmm.fused_brgemm(data_type = f32, %0, %arg0, %arg0, %arg1, %arg2, %0) :
    (i64, memref<1x3x3xf32>, memref<1x3x3xf32>, memref<3x3xf32>, memref<3xf32>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.fused_brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] [add, relu] 
    flags = (none) binary_flags = (none) unary_flags = (none) data_type = bf16
  // expected-error@+1 {{inconsistent data types}}
  xsmm.fused_brgemm(data_type = f32, %0, %arg0, %arg0, %arg1, %arg1, %0) :
    (i64, memref<1x3x3xf32>, memref<1x3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xbf16>, %arg1: memref<3x3xbf16>) {
  %0 = xsmm.fused_brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] [add, relu] 
    flags = (vnni_a) binary_flags = (none) unary_flags = (none) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand A or invalid VNNI_A flags}}
  xsmm.fused_brgemm(data_type = bf16, %0, %arg0, %arg0, %arg1, %arg1, %0) :
    (i64, memref<1x3x3xbf16>, memref<1x3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xbf16>, %arg1: memref<3x3xbf16>) {
  %0 = xsmm.fused_brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] [add, relu] 
    flags = (vnni_b) binary_flags = (none) unary_flags = (none) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand B or invalid VNNI_B flags}}
  xsmm.fused_brgemm(data_type = bf16, %0, %arg0, %arg0, %arg1, %arg1, %0) :
    (i64, memref<1x3x3xbf16>, memref<1x3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>, i64) -> ()
  return
}

// -----

func.func @brgemm(%arg0: memref<1x3x3xbf16>, %arg1: memref<3x3xbf16>) {
  %0 = xsmm.fused_brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] [add, relu] 
    flags = (vnni_c) binary_flags = (none) unary_flags = (none) data_type = bf16
  // expected-error@+1 {{expect VNNI layout for operand C or invalid VNNI_C flags}}
  xsmm.fused_brgemm(data_type = bf16, %0, %arg0, %arg0, %arg1, %arg1, %0) :
    (i64, memref<1x3x3xbf16>, memref<1x3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>, i64) -> ()
  return
}

// -----

func.func @unary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (none) data_type = f32
  // expected-error@+1 {{invalid dispatch operation}}
  xsmm.unary relu(data_type = f32, %0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @unary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.unary.dispatch relu [3, 3, 3, 3] flags = (none) data_type = bf16
  // expected-error@+1 {{inconsistent data types}}
  xsmm.unary relu(data_type = f32, %0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @unary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.unary.dispatch identity [3, 3, 3, 3] flags = (none) data_type = f32
  // expected-error@+1 {{inconsistent callee kind}}
  xsmm.unary relu(data_type = f32, %0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @unary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.unary.dispatch relu [3, 3, 3, 3] flags = (bcast_scalar) data_type = f32
  // expected-error@+1 {{invalid 'bcast_scalar' flag for input}}
  xsmm.unary relu(data_type = f32, %0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.brgemm.dispatch [3, 3, 3, 3, 3, 3, 1, 1] flags = (none) data_type = f32
  // expected-error@+1 {{invalid dispatch operation}}
  xsmm.binary add(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.binary.dispatch add [3, 3, 3, 3, 3] flags = (none) data_type = bf16
  // expected-error@+1 {{inconsistent data types}}
  xsmm.binary add(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.binary.dispatch sub [3, 3, 3, 3, 3] flags = (none) data_type = f32
  // expected-error@+1 {{inconsistent callee kind}}
  xsmm.binary add(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.binary.dispatch add [3, 3, 3, 3, 3] flags = (bcast_scalar_in0) data_type = f32
  // expected-error@+1 {{invalid 'bcast_scalar_in0' flag for lhs input}}
  xsmm.binary add(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %0 = xsmm.binary.dispatch add [3, 3, 3, 3, 3] flags = (bcast_scalar_in1) data_type = f32
  // expected-error@+1 {{invalid 'bcast_scalar_in1' flag for rhs input}}
  xsmm.binary add(data_type = f32, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}
