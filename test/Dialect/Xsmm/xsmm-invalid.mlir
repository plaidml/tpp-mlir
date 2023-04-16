// RUN: tpp-opt -split-input-file -verify-diagnostics %s

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_b) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{expected flags to be unique}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{'none' flags conflicts with others}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (none, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{expect 6 args but got: 5}}
  %0 = xsmm.matmul.dispatch [3, 2, 1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @matmul_dispatch
func.func @matmul_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.matmul.dispatch [-3, 2, 1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_b) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_c) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{expected flags to be unique}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{'none' flags conflicts with others}}
  %0 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (none, vnni_a) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @brgemm_dispatch
func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{expect 6 args but got: 5}}
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
  %0 = xsmm.unary.dispatch relu [3, 2, 1, -3] ( broadcast none dataType f32 )
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @unary_dispatch
func.func @unary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 4 args but got: 3}}
  %0 = xsmm.unary.dispatch relu [3, 2, 1] ( broadcast none dataType f32 )
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @binary_dispatch
func.func @binary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.binary.dispatch add [3, 2, 1, 3, -2] ( broadcast none dataType f32 )
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @binary_dispatch
func.func @binary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 5 args but got: 3}}
  %0 = xsmm.binary.dispatch add [3, 2, 1] ( broadcast none dataType f32 )
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @ternary_dispatch
func.func @ternary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.ternary.dispatch none [3, 2, 1, 3, -2] ( dataType f32 )
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @ternary_dispatch
func.func @ternary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 6 args but got: 3}}
  %0 = xsmm.ternary.dispatch none [3, 2, 1] ( dataType f32 )
  return %0 : i64
}
