// RUN: tpp-opt -split-input-file -verify-diagnostics %s

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_b) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{expected flags to be unique}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{'none' flags conflicts with others}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2, 1] flags = (none, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{expect 6 args but got: 5}}
  %0 = xsmm.gemm.dispatch [3, 2, 1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @gemm_dispatch
func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.gemm.dispatch [-3, 2, 1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1, 1, 1] flags = (vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1, 1, 1] flags = (vnni_b) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1, 1, 1] flags = (vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1, 1, 1] flags = (vnni_a, vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{expected flags to be unique}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1, 1, 1] flags = (vnni_a, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{'none' flags conflicts with others}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1, 1, 1] flags = (none, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{expect 8 args but got: 5}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.brgemm.dispatch [3, 2, -1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @unary_dispatch
func.func @unary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.unary.dispatch relu [3, 2, 1, -3] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @unary_dispatch
func.func @unary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 4 args but got: 3}}
  %0 = xsmm.unary.dispatch relu [3, 2, 1] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @binary_dispatch
func.func @binary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.binary.dispatch add [3, 2, 1, 3, -2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @binary_dispatch
func.func @binary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 5 args but got: 3}}
  %0 = xsmm.binary.dispatch add [3, 2, 1] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @ternary_dispatch
func.func @ternary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.ternary.dispatch none [3, 2, 1, 3, -2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @ternary_dispatch
func.func @ternary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 6 args but got: 3}}
  %0 = xsmm.ternary.dispatch none [3, 2, 1] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @fused_dispatch
func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expect 8 args but got: 3}}
  %0 = xsmm.fused_brgemm.dispatch [3, 2, 1] [add, relu] 
    flags = (none) binary_flags = (none) unary_flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @fused_dispatch
func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expected flags to be unique}}
  %0 = xsmm.fused_brgemm.dispatch [3, 2, 1, 1, 1, 1, 1, 1] [add, relu] 
    flags = (vnni_a, vnni_a) binary_flags = (none) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @fused_dispatch
func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expected binary_flags to be unique}}
  %0 = xsmm.fused_brgemm.dispatch [3, 2, 1, 1, 1, 1] [add, relu] 
    flags = (vnni_a) binary_flags = (none, none) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @fused_dispatch
func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expected unary_flags to be unique}}
  %0 = xsmm.fused_brgemm.dispatch [3, 2, 1, 1, 1, 1] [add, relu] 
    flags = (vnni_a) binary_flags = (none) unary_flags = (none, none) data_type = bf16
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @fused_brgemm_none_kind_with_flags
func.func @fused_brgemm_none_kind_with_flags() -> i64 {
  // expected-error@+1 {{invalid binary flags for kind none}}
   %0 = xsmm.fused_brgemm.dispatch [3, 2, 1, 1, 1, 1, 1, 1] [none, relu]
    flags = (vnni_a) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @fused_brgemm_none_kind_with_flags
func.func @fused_brgemm_none_kind_with_flags() -> i64 {
  // expected-error@+1 {{invalid unary flags for kind none}}
   %0 = xsmm.fused_brgemm.dispatch [3, 2, 1, 1, 1, 1, 1, 1] [none, none]
    flags = (vnni_a) binary_flags = (none) unary_flags = (bcast_scalar) data_type = bf16
  return %0 : i64
}
