// RUN: standalone-opt %s -split-input-file -linalg-ext-to-loops | FileCheck %s

func.func @NC_to_NCnc(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  linalgx.pack %arg0 dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : (memref<128x256xf32> memref<4x8x32x32xf32>)
  return
}
// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @NC_to_NCnc(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ubN:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[block:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[N:.*]] = %[[lb]] to %[[ubN]] step %[[step]] {
// CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:     scf.for %[[n:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:       scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK-DAG:         %[[applyMapI:.*]] = affine.apply #[[MAP]](%[[N]], %[[n]])
// CHECK-DAG:         %[[applyMapJ:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
// CHECK:         %[[scalar:.*]] = memref.load %arg0[%[[applyMapI]], %[[applyMapJ]]] : memref<128x256xf32>
// CHECK:         memref.store %[[scalar]], %arg1[%[[N]], %[[C]], %[[n]], %[[c]]] : memref<4x8x32x32xf32>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @NC_to_NCnc_pad_static(%arg0: memref<13x15xf32>, %arg1: memref<2x8x8x2xf32>, %arg2: f32) {
  linalgx.pack %arg0 padding_value(%arg2 : f32) dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : (memref<13x15xf32> memref<2x8x8x2xf32>)
  return
}
// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL: func.func @NC_to_NCnc_pad_static(
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[C13:.*]] = arith.constant 13 : index
// CHECK-DAG:     %[[C15:.*]] = arith.constant 15 : index
// CHECK:           scf.for %[[N:.*]] = %[[C0]] to %[[C2]] step %[[step]] {
// CHECK:             scf.for %[[C:.*]] = %[[C0]] to %[[C8]] step %[[step]] {
// CHECK:               scf.for %[[n:.*]] = %[[C0]] to %[[C8]] step %[[step]] {
// CHECK:                 scf.for %[[c:.*]] = %[[C0]] to %[[C2]] step %[[step]] {
// CHECK-DAG:               %[[applyMapI:.*]] = affine.apply #[[MAP0]](%[[N]], %[[n]])
// CHECK-DAG:               %[[applyMapJ:.*]] = affine.apply #[[MAP1]](%[[C]], %[[c]])
// CHECK:                   %[[isIInBound:.*]] = arith.cmpi slt, %[[applyMapI]], %[[C13]] : index
// CHECK:                   %[[isJInBound:.*]] = arith.cmpi slt, %[[applyMapJ]], %[[C15]] : index
// CHECK:                   %[[isAllInBounds:.*]] = arith.andi %[[isIInBound]], %[[isJInBound]] : i1
// CHECK:                   %[[scalar:.*]] = scf.if %[[isAllInBounds]] -> (f32) {
// CHECK:                     %[[load:.*]] = memref.load %arg0[%[[applyMapI]], %[[applyMapJ]]] : memref<13x15xf32>
// CHECK:                     scf.yield %[[load]]
// CHECK:                   } else {
// CHECK:                     scf.yield %arg2
// CHECK:                   }
// CHECK:                   memref.store %[[scalar]], %arg1[%[[N]], %[[C]], %[[n]], %[[c]]] : memref<2x8x8x2xf32>

// -----

func.func @KC_to_KCck(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  linalgx.pack %arg0 dims_pos = [1, 0] inner_tiles = [32, 32] into %arg1 : (memref<128x256xf32> memref<4x8x32x32xf32>)
  return
}
// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @KC_to_KCck(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ubK:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[block:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
// CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:     scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:       scf.for %[[k:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK-DAG:         %[[applyMapC:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
// CHECK-DAG:         %[[applyMapK:.*]] = affine.apply #[[MAP]](%[[K]], %[[k]])
// CHECK:         %[[scalar:.*]] = memref.load %arg0[%[[applyMapK]], %[[applyMapC]]] : memref<128x256xf32>
// CHECK:         memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[c]], %[[k]]] : memref<4x8x32x32xf32>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

// This should be a simple expand shape.
func.func @KC_to_KCc(%arg0: memref<128x256xf32>, %arg1: memref<128x8x32xf32>) {
  linalgx.pack %arg0 dims_pos = [1] inner_tiles = [32] into %arg1 : (memref<128x256xf32> memref<128x8x32xf32>)
  return
}
// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @KC_to_KCc(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubK:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[block:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
// CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:     scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:       %[[applyMapC:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
// CHECK:       %[[scalar:.*]] = memref.load %arg0[%[[K]], %[[applyMapC]]] : memref<128x256xf32>
// CHECK:       memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[c]]] : memref<128x8x32xf32>
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @KC_to_KCk(%arg0: memref<128x256xf32>, %arg1: memref<4x256x32xf32>) {
  linalgx.pack %arg0 dims_pos = [0] inner_tiles = [32] into %arg1 : (memref<128x256xf32> memref<4x256x32xf32>)
  return
}

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-LABEL: func.func @KC_to_KCk(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubC:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[ubK:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[block:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
// CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
// CHECK:     scf.for %[[k:.*]] = %[[lb]] to %[[block]] step %[[step]] {
// CHECK:       %[[applyMapK:.*]] = affine.apply #[[MAP]](%[[K]], %[[k]])
// CHECK:       %[[scalar:.*]] = memref.load %arg0[%[[applyMapK]], %[[C]]] : memref<128x256xf32>
// CHECK:       memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[k]]] : memref<4x256x32xf32>
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @KCRS_to_KCRSck(%arg0: memref<128x64x1x1xf32>, %arg1: memref<4x8x1x1x8x32xf32>) {
  linalgx.pack %arg0 dims_pos = [1, 0] inner_tiles = [8, 32] into %arg1 : (memref<128x64x1x1xf32> memref<4x8x1x1x8x32xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-LABEL: func.func @KCRS_to_KCRSck(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubK:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[ubC:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[blockK:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[one]] {
// CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[one]] {
// CHECK:     scf.for %[[R:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:       scf.for %[[S:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:         scf.for %[[c:.*]] = %[[lb]] to %[[ubC]] step %[[one]] {
// CHECK:           scf.for %[[k:.*]] = %[[lb]] to %[[blockK]] step %[[one]] {
// CHECK-DAG:         %[[affineMapK:.*]] = affine.apply #[[MAP0]](%[[K]], %[[k]])
// CHECK-DAG:         %[[affineMapC:.*]] = affine.apply #[[MAP1]](%[[C]], %[[c]])
// CHECK:             %[[scalar:.*]] = memref.load %arg0[%[[affineMapK]], %[[affineMapC]], %[[R]], %[[S]]] : memref<128x64x1x1xf32>
// CHECK:             memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[R]], %[[S]], %[[c]], %[[k]]] : memref<4x8x1x1x8x32xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @KCRS_to_KCRSsr(%arg0: memref<1x1x128x64xf32>, %arg1: memref<1x1x4x8x8x32xf32>) {
  linalgx.pack %arg0 dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (memref<1x1x128x64xf32> memref<1x1x4x8x8x32xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-LABEL: func.func @KCRS_to_KCRSsr(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubR:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[ubS:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[blockR:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[K:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[one]] step %[[one]] {
// CHECK:     scf.for %[[R:.*]] = %[[lb]] to %[[ubR]] step %[[one]] {
// CHECK:       scf.for %[[S:.*]] = %[[lb]] to %[[ubS]] step %[[one]] {
// CHECK:         scf.for %[[s:.*]] = %[[lb]] to %[[ubS]] step %[[one]] {
// CHECK:           scf.for %[[r:.*]] = %[[lb]] to %[[blockR]] step %[[one]] {
// CHECK-DAG:         %[[affineMapR:.*]] = affine.apply #[[MAP0]](%[[R]], %[[r]])
// CHECK-DAG:         %[[affineMapS:.*]] = affine.apply #[[MAP1]](%[[S]], %[[s]])
// CHECK:             %[[scalar:.*]] = memref.load %arg0[%[[K]], %[[C]], %[[affineMapR]], %[[affineMapS]]] : memref<1x1x128x64xf32>
// CHECK:             memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[R]], %[[S]], %[[s]], %[[r]]] : memref<1x1x4x8x8x32xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

// Test to check that we properly handle shuffled `dims_pos` and `tiles.
// In this example, the dimension at position `0` (aka `128`) is tiled with a factor of `32`.
// While the dimension at position `2` (aka `2`) is tiled with a factor of `2`.
func.func @shuffled_dim_pos_and_tiles(%arg0: memref<128x256x2x1000xf32>, %arg1: memref<4x256x1x1000x2x32xf32>) {
  linalgx.pack %arg0 dims_pos = [2, 0] inner_tiles = [2, 32] into %arg1 : (memref<128x256x2x1000xf32> memref<4x256x1x1000x2x32xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL: func.func @shuffled_dim_pos_and_tiles(
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubDimZero:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[ubDimOne:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[ubDimThree:.*]] = arith.constant 1000 : index
// CHECK-DAG: %[[ubDimFour:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[ubDimFive:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ubDimZero]] step %[[step]] {
// CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ubDimOne]] step %[[step]] {
// CHECK:     scf.for %[[k:.*]] = %[[lb]] to %[[step]] step %[[step]] {
// CHECK:       scf.for %[[l:.*]] = %[[lb]] to %[[ubDimThree]] step %[[step]] {
// CHECK:         scf.for %[[m:.*]] = %[[lb]] to %[[ubDimFour]] step %[[step]] {
// CHECK:           scf.for %[[n:.*]] = %[[lb]] to %[[ubDimFive]] step %[[step]] {
// CHECK-DAG:         %[[affineApplyZero:.*]] = affine.apply #[[MAP0]](%[[i]], %[[n]])
// CHECK-DAG:         %[[affineApplyOne:.*]] = affine.apply #[[MAP1]](%[[k]], %[[m]])
// CHECK:             %[[scalar:.*]] = memref.load %arg0[%[[affineApplyZero]], %[[j]], %[[affineApplyOne]], %[[l]]] : memref<128x256x2x1000xf32>
// CHECK:             memref.store %[[scalar]], %arg1[%[[i]], %[[j]], %[[k]], %[[l]], %[[m]], %[[n]]] : memref<4x256x1x1000x2x32xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @KCRS_to_KCRSsr(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x32xf32>) {
  linalgx.pack %arg0 dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : (memref<?x?x?x?xf32> memref<?x?x?x?x8x32xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 ceildiv 32)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK-LABEL: func.func @KCRS_to_KCRSsr(
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[eight:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[thirtytwo:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[two:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[three:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[dimZero:.*]] = memref.dim %arg0, %[[zero]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[dimOne:.*]] = memref.dim %arg0, %[[one]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[dimTwo:.*]] = memref.dim %arg0, %[[two]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[dimThree:.*]] = memref.dim %arg0, %[[three]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[mapOnDimTwo:.*]] = affine.apply #[[MAP0]]()[%[[dimTwo]]]
// CHECK-DAG: %[[mapOnDimThree:.*]] = affine.apply #[[MAP1]]()[%[[dimThree]]]
// CHECK: scf.for %[[K:.*]] = %[[zero]] to %[[dimZero]] step %[[one]] {
// CHECK:   scf.for %[[C:.*]] = %[[zero]] to %[[dimOne]] step %[[one]] {
// CHECK:     scf.for %[[R:.*]] = %[[zero]] to %[[mapOnDimTwo]] step %[[one]] {
// CHECK:       scf.for %[[S:.*]] = %[[zero]] to %[[mapOnDimThree]] step %[[one]] {
// CHECK:         scf.for %[[s:.*]] = %[[zero]] to %[[eight]] step %[[step]] {
// CHECK:           scf.for %[[r:.*]] = %[[zero]] to %[[thirtytwo]] step %[[step]] {
// CHECK-DAG:         %[[affineMapR:.*]] = affine.apply #[[MAP2]](%[[R]], %[[r]])
// CHECK-DAG:         %[[affineMapS:.*]] = affine.apply #[[MAP3]](%[[S]], %[[s]])
// CHECK:             %[[scalar:.*]] = memref.load %arg0[%[[K]], %[[C]], %[[affineMapR]], %[[affineMapS]]] : memref<?x?x?x?xf32>
// CHECK:             memref.store %[[scalar]], %arg1[%[[K]], %[[C]], %[[R]], %[[S]], %[[s]], %[[r]]] : memref<?x?x?x?x8x32xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @KCRS_to_KCRSsr(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x?xf32>, %block : index) {
  linalgx.pack %arg0 dims_pos = [3, 2] inner_tiles = [8, %block] into %arg1 : (memref<?x?x?x?xf32> memref<?x?x?x?x8x?xf32>)
  return
}

// CHECK-DAG:  #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
// CHECK-DAG:  #[[MAP1:.*]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
// CHECK-DAG:  #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK:      func.func @KCRS_to_KCRSsr
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[eight:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[two:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[three:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[dimZero:.*]] = memref.dim %[[ARG0]], %[[zero]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[dimOne:.*]] = memref.dim %[[ARG0]], %[[one]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[dimTwo:.*]] = memref.dim %[[ARG0]], %[[two]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[dimThree:.*]] = memref.dim %[[ARG0]], %[[three]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[mapOnDimTwo:.*]] = affine.apply #[[MAP0]]()[%[[dimTwo]], %[[ARG2]]]
// CHECK-DAG: %[[mapOnDimThree:.*]] = affine.apply #[[MAP1]]()[%[[dimThree]]]
// CHECK: scf.for %[[K:.*]] = %[[zero]] to %[[dimZero]] step %[[one]] {
// CHECK:   scf.for %[[C:.*]] = %[[zero]] to %[[dimOne]] step %[[one]] {
// CHECK:     scf.for %[[R:.*]] = %[[zero]] to %[[mapOnDimTwo]] step %[[one]] {
// CHECK:       scf.for %[[S:.*]] = %[[zero]] to %[[mapOnDimThree]] step %[[one]] {
// CHECK:         scf.for %[[s:.*]] = %[[zero]] to %[[eight]] step %[[step]] {
// CHECK:           scf.for %[[r:.*]] = %[[zero]] to %[[ARG2]] step %[[step]] {
// CHECK-DAG:         %[[affineMapR:.*]] = affine.apply #[[MAP2]](%[[R]], %[[r]])
// CHECK-DAG:         %[[affineMapS:.*]] = affine.apply #[[MAP3]](%[[S]], %[[s]])
// CHECK:             %[[scalar:.*]] = memref.load %[[ARG0]][%[[K]], %[[C]], %[[affineMapR]], %[[affineMapS]]] : memref<?x?x?x?xf32>
// CHECK:             memref.store %[[scalar]], %[[ARG1]][%[[K]], %[[C]], %[[R]], %[[S]], %[[s]], %[[r]]] : memref<?x?x?x?x8x?xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// -----

func.func @KCRSsr_to_KCRS(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x32xf32>) {
  linalgx.unpack %arg1 dims_pos = [3, 2] inner_tiles = [8, 32] into %arg0 : (memref<?x?x?x?x8x32xf32> memref<?x?x?x?xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOORK:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG: #[[MAP_MODK:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG: #[[MAP_FLOORL:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG: #[[MAP_MODL:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK: func.func @KCRSsr_to_KCRS
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[UBI:.*]] = memref.dim %[[ARG0]], %[[ZERO]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[UBJ:.*]] = memref.dim %[[ARG0]], %[[ONE]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[UBK:.*]] = memref.dim %[[ARG0]], %[[TWO]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[UBL:.*]] = memref.dim %[[ARG0]], %[[THREE]] : memref<?x?x?x?xf32>
// CHECK: scf.for %[[I:.*]] = %[[ZERO]] to %[[UBI]] step %[[ONE]] {
// CHECK: scf.for %[[J:.*]] = %[[ZERO]] to %[[UBJ]] step %[[ONE]] {
// CHECK: scf.for %[[K:.*]] = %[[ZERO]] to %[[UBK]] step %[[ONE]] {
// CHECK: scf.for %[[L:.*]] = %[[ZERO]] to %[[UBL]] step %[[ONE]] {
// CHECK-DAG: %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])
// CHECK-DAG: %[[FLOORL:.*]] = affine.apply #[[MAP_FLOORL]](%[[L]])
// CHECK-DAG: %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])
// CHECK-DAG: %[[MODL:.*]] = affine.apply #[[MAP_MODL]](%[[L]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[J]], %[[FLOORK]], %[[FLOORL]], %[[MODL]], %[[MODK]]] : memref<?x?x?x?x8x32xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<?x?x?x?xf32>
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }

// -----

func.func @KCRSsr_to_KCRS(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?x8x?xf32>, %block : index) {
  linalgx.unpack %arg1 dims_pos = [3, 2] inner_tiles = [8, %block] into %arg0 : (memref<?x?x?x?x8x?xf32> memref<?x?x?x?xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOORK:.*]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK-DAG: #[[MAP_MODK:.*]] = affine_map<(d0)[s0] -> (d0 mod s0)>
// CHECK-DAG: #[[MAP_FLOORL:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG: #[[MAP_MODL:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK: func.func @KCRSsr_to_KCRS
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[UBI:.*]] = memref.dim %[[ARG0]], %[[ZERO]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[UBJ:.*]] = memref.dim %[[ARG0]], %[[ONE]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[UBK:.*]] = memref.dim %[[ARG0]], %[[TWO]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[UBL:.*]] = memref.dim %[[ARG0]], %[[THREE]] : memref<?x?x?x?xf32>
// CHECK: scf.for %[[I:.*]] = %[[ZERO]] to %[[UBI]] step %[[ONE]] {
// CHECK: scf.for %[[J:.*]] = %[[ZERO]] to %[[UBJ]] step %[[ONE]] {
// CHECK: scf.for %[[K:.*]] = %[[ZERO]] to %[[UBK]] step %[[ONE]] {
// CHECK: scf.for %[[L:.*]] = %[[ZERO]] to %[[UBL]] step %[[ONE]] {
// CHECK-DAG: %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])[%[[ARG2]]]
// CHECK-DAG: %[[FLOORL:.*]] = affine.apply #[[MAP_FLOORL]](%[[L]])
// CHECK-DAG: %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])[%[[ARG2]]]
// CHECK-DAG: %[[MODL:.*]] = affine.apply #[[MAP_MODL]](%[[L]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[J]], %[[FLOORK]], %[[FLOORL]], %[[MODL]], %[[MODK]]] : memref<?x?x?x?x8x?xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<?x?x?x?xf32>
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }

// -----

func.func @KCRSsr_to_KCRS(%arg0: memref<1x1x128x64xf32>, %arg1: memref<1x1x4x8x8x32xf32>) {
  linalgx.unpack %arg1 dims_pos = [3, 2] inner_tiles = [8, 32] into %arg0 : (memref<1x1x4x8x8x32xf32> memref<1x1x128x64xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOORK:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG: #[[MAP_MODK:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG: #[[MAP_FLOORL:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG: #[[MAP_MODL:.*]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK: func.func @KCRSsr_to_KCRS
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[UBK:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[UBL:.*]] = arith.constant 64 : index
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[STEP]] step %[[STEP]] {
// CHECK: scf.for %[[J:.*]] = %[[LB]] to %[[STEP]] step %[[STEP]] {
// CHECK: scf.for %[[K:.*]] = %[[LB]] to %[[UBK]] step %[[STEP]] {
// CHECK: scf.for %[[L:.*]] = %[[LB]] to %[[UBL]] step %[[STEP]] {
// CHECK-DAG: %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])
// CHECK-DAG: %[[FLOORL:.*]] = affine.apply #[[MAP_FLOORL]](%[[L]])
// CHECK-DAG: %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])
// CHECK-DAG: %[[MODL:.*]] = affine.apply #[[MAP_MODL]](%[[L]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[J]], %[[FLOORK]], %[[FLOORL]], %[[MODL]], %[[MODK]]] : memref<1x1x4x8x8x32xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<1x1x128x64xf32>
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }

// -----

func.func @shuffled_dim_pos_and_tiles(%arg0: memref<128x256x2x1000xf32>, %arg1: memref<4x256x1x1000x2x32xf32>) {
  linalgx.unpack %arg1 dims_pos = [2, 0] inner_tiles = [2, 32] into %arg0 : (memref<4x256x1x1000x2x32xf32> memref<128x256x2x1000xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOORI:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG: #[[MAP_MODI:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG: #[[MAP_FLOORK:.*]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG: #[[MAP_MODK:.*]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK: func.func @shuffled_dim_pos_and_tiles
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[UBJ:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[UBK:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[UBL:.*]] = arith.constant 1000 : index
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK: scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK: scf.for %[[K:.*]] = %[[LB]] to %[[UBK]] step %[[STEP]] {
// CHECK: scf.for %[[L:.*]] = %[[LB]] to %[[UBL]] step %[[STEP]] {
// CHECK-DAG: %[[FLOORI:.*]] = affine.apply #[[MAP_FLOORI]](%[[I]])
// CHECK-DAG: %[[MODI:.*]] = affine.apply #[[MAP_MODI]](%[[I]])
// CHECK-DAG: %[[FLOORK:.*]] = affine.apply #[[MAP_FLOORK]](%[[K]])
// CHECK-DAG: %[[MODK:.*]] = affine.apply #[[MAP_MODK]](%[[K]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[FLOORI]], %[[J]], %[[FLOORK]], %[[L]], %[[MODK]], %[[MODI]]] : memref<4x256x1x1000x2x32xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]], %[[K]], %[[L]]] : memref<128x256x2x1000xf32>
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }

// -----

func.func @KCck_to_KC(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  linalgx.unpack %arg1 dims_pos = [1, 0] inner_tiles = [32, 32] into %arg0 : (memref<4x8x32x32xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOOR:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG: #[[MAP_MOD:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK: func.func @KCck_to_KC
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[UBJ:.*]] = arith.constant 256 : index
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK: scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK-DAG: %[[FLOORI:.*]] = affine.apply #[[MAP_FLOOR]](%[[I]])
// CHECK-DAG: %[[FLOORJ:.*]] = affine.apply #[[MAP_FLOOR]](%[[J]])
// CHECK-DAG: %[[MODI:.*]] = affine.apply #[[MAP_MOD]](%[[I]])
// CHECK-DAG: %[[MODJ:.*]] = affine.apply #[[MAP_MOD]](%[[J]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[FLOORI]], %[[FLOORJ]], %[[MODJ]], %[[MODI]]] : memref<4x8x32x32xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]]] : memref<128x256xf32>
// CHECK: }
// CHECK: }

// -----

// This should be a simple collapse shape.
func.func @KCc_to_KC(%arg0: memref<128x256xf32>, %arg1: memref<128x8x32xf32>) {
  linalgx.unpack %arg1 dims_pos = [1] inner_tiles = [32] into %arg0 : (memref<128x8x32xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOOR:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG: #[[MAP_MOD:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK: func.func @KCc_to_KC
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[UBJ:.*]] = arith.constant 256 : index
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK: scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK-DAG: %[[FLOORJ:.*]] = affine.apply #[[MAP_FLOOR]](%[[J]])
// CHECK-DAG: %[[MODJ:.*]] = affine.apply #[[MAP_MOD]](%[[J]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[I]], %[[FLOORJ]], %[[MODJ]]] : memref<128x8x32xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]]] : memref<128x256xf32>
// CHECK: }
// CHECK: }

// -----

func.func @NCnc_to_NC(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  linalgx.unpack %arg1 dims_pos = [0, 1] inner_tiles = [32, 32] into %arg0 : (memref<4x8x32x32xf32> memref<128x256xf32>)
  return
}

// CHECK-DAG: #[[MAP_FLOOR:.*]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG: #[[MAP_MOD:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK: func.func @NCnc_to_NC
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG: %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[UBI:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[UBJ:.*]] = arith.constant 256 : index
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UBI]] step %[[STEP]] {
// CHECK: scf.for %[[J:.*]] = %[[LB]] to %[[UBJ]] step %[[STEP]] {
// CHECK-DAG: %[[FLOORI:.*]] = affine.apply #[[MAP_FLOOR]](%[[I]])
// CHECK-DAG: %[[FLOORJ:.*]] = affine.apply #[[MAP_FLOOR]](%[[J]])
// CHECK-DAG: %[[MODI:.*]] = affine.apply #[[MAP_MOD]](%[[I]])
// CHECK-DAG: %[[MODJ:.*]] = affine.apply #[[MAP_MOD]](%[[J]])
// CHECK: %[[SCALAR:.*]] = memref.load %[[ARG1]][%[[FLOORI]], %[[FLOORJ]], %[[MODI]], %[[MODJ]]] : memref<4x8x32x32xf32>
// CHECK: memref.store %[[SCALAR]], %[[ARG0]][%[[I]], %[[J]]] : memref<128x256xf32>
// CHECK: }
// CHECK: }
