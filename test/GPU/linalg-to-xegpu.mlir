// RUN: tpp-opt %s -linalg-to-xegpu=k-tile=16 -canonicalize -split-input-file | FileCheck %s

func.func @matmul(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul
// CHECK-COUNT-3: xegpu.load_nd
// CHECK: xegpu.dpas
// CHECH: xegpu.store_nd

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @generic_matmul(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf16>) {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>) outs(%arg2 : memref<8x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.mulf %in, %in_0 : f16
      %1 = arith.addf %out, %0 : f16
      linalg.yield %1 : f16
    }
    return
  }
}

// CHECK-LABEL: func.func @generic_matmul
// CHECK-COUNT-3: xegpu.load_nd
// CHECK: xegpu.dpas
// CHECH: xegpu.store_nd

// -----

func.func @matmul_trunc_result(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @matmul_trunc_result
// CHECK: arith.extf
// CHECK: xegpu.dpas
// CHECK: arith.truncf
// CHECH: xegpu.store_nd

// -----

func.func @abs(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.abs ins(%arg0 : memref<8x16xf16>)
             outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @abs
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.absf
// CHECK: xegpu.store_nd

// -----

func.func @add(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.add ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @add
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.addf
// CHECK: xegpu.store_nd

// -----

func.func @ceil(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.ceil ins(%arg0 : memref<8x16xf16>)
              outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @ceil
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.ceil
// CHECK: xegpu.store_nd

// -----

func.func @div(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.div ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @div
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.divf
// CHECK: xegpu.store_nd

// -----

func.func @div_unsigned(%arg0: memref<8x16xi16>, %arg1: memref<8x16xi16>, %arg2: memref<8x16xi16>) {
  linalg.div_unsigned ins(%arg0, %arg1 : memref<8x16xi16>, memref<8x16xi16>)
                      outs(%arg2 : memref<8x16xi16>)
  return
}

// CHECK-LABEL: func.func @div_unsigned
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.divui
// CHECK: xegpu.store_nd

// -----

func.func @exp(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.exp ins(%arg0 : memref<8x16xf16>)
             outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @exp
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.exp
// CHECK: xegpu.store_nd

// -----

func.func @floor(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.floor ins(%arg0 : memref<8x16xf16>)
               outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @floor
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.floor
// CHECK: xegpu.store_nd

// -----

func.func @max(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.max ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @max
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.maximumf
// CHECK: xegpu.store_nd

// -----

func.func @mul(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.mul ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @mul
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.mulf
// CHECK: xegpu.store_nd

// -----

func.func @negf(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.negf ins(%arg0 : memref<8x16xf16>)
              outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @negf
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: arith.negf
// CHECK: xegpu.store_nd

// -----

func.func @sub(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.sub ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @sub
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.subf
// CHECK: xegpu.store_nd
