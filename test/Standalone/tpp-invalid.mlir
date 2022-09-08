// RUN: standalone-opt %s -split-input-file -verify-diagnostics 

func.func @myfunc(%arg0: memref<1x2xf32>, 
                  %arg1: memref<2x2xf32>, %arg2: memref<2x1xf32>) -> memref<2x1xf32> {

  // expected-error @below {{'tpp.add' op requires all operands to have the same type}}
  tpp.add ins(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) 
          out(%arg2: memref<2x1xf32>)

  return %arg2: memref<2x1xf32>
}

// -----

func.func @myfunc(%arg0: memref<1x2xf32>, %arg1: memref<2x1xf32>) -> memref<2x1xf32> {

  // expected-error @below {{'tpp.relu' op requires all operands to have the same type}}
  tpp.relu ins(%arg0: memref<1x2xf32>) out(%arg1: memref<2x1xf32>)
  return %arg1: memref<2x1xf32>
}

// -----

func.func @myfunc(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) -> memref<1x2xf32> {

  // expected-error @below {{broadcast incompatible}}
  tpp.identity ins(%arg1: memref<2x2xf32>) out(%arg0: memref<1x2xf32>)
  return %arg0: memref<1x2xf32>
}

// -----

func.func @myfunc(%arg0: memref<?x?xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{'tpp.identity' op operand #0 must be 1D/2D memref of floating-point values or floating-point, but got 'memref<?x?xf32>'}}
  tpp.identity ins(%arg0: memref<?x?xf32>) out(%arg1: memref<2x2xf32>)
  return %arg1: memref<2x2xf32>
}

// -----

func.func @myfunc(%arg0: memref<3x3xf32>, %arg1: memref<2x3xf32>) -> memref<3x3xf32> {

  // expected-error @below {{broadcast incompatible}}
  tpp.identity ins(%arg1: memref<2x3xf32>) out(%arg0: memref<3x3xf32>)
  return %arg0: memref<3x3xf32>
}
