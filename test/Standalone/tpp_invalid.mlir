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

func.func @myfunc(%arg0: memref<2x2xf32>, %arg1: memref<1x2xf32>) -> memref<2x2xf32> {

  // expected-error @below {{incompatible shape}}
  tpp.identity ins(%arg1: memref<1x2xf32>) out(%arg0: memref<2x2xf32>)
  return %arg0: memref<2x2xf32>
}
