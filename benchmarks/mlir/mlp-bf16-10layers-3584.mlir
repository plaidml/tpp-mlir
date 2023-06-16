// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) + BiasAdd (n*m) + ReLU (O(n*m) x 10
// ( 2*256x3584x3584 (6576668672) + 256x3584 (917504) + 256x3584 (917504) ) x 10 = 65,785,036,800
// BENCH_TOTAL_FLOPS: 65785036800

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

func.func @entry(%arg0: tensor<8x112x32x32xbf16>) -> tensor<8x112x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16

  // Zero initialized output buffer for GEMMs
  %zero = arith.constant 0.000000e+00 : bf16
  %init_shape = tensor.empty() : tensor<8x112x32x32xbf16>
  %zero_init = linalg.fill ins(%zero : bf16) outs(%init_shape : tensor<8x112x32x32xbf16>) -> tensor<8x112x32x32xbf16> 

  // Use the same weights and biases for all the layers due to the extremely high memory
  // usage of MLIR dense constants
  // TODO: replace with different constants for each layer when memory issues are fixed
  %weights = arith.constant dense<0.01> : tensor<112x112x32x32xbf16>
  %bias = arith.constant dense<0.02> : tensor<3584xbf16>

  // Empty tensor to indicate shape of the output buffer
  %out_shape = tensor.empty() : tensor<8x112x32x32xbf16>

  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%0, %expanded : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%2 : tensor<8x112x32x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%3, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded2 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%4, %expanded2 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%6 : tensor<8x112x32x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%7, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded3 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %10 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%8, %expanded3 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%10 : tensor<8x112x32x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

   %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%11, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded4 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %14 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%12, %expanded4 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %15 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%14 : tensor<8x112x32x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  %16 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%15, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded5 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %18 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%16, %expanded5 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %19 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%18 : tensor<8x112x32x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

   %20 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%19, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded6 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %22 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%20, %expanded6 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %23 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%22 : tensor<8x112x32x32xbf16>)
    outs(%out_shape : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  %24 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%23, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded7 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %26 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%24, %expanded7 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %27 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%26 : tensor<8x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>


  %28 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%27, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded8 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %30 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%28, %expanded8 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %31 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%30 : tensor<8x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  %32 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%31, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded9 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %34 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%32, %expanded9 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %35 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%34 : tensor<8x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  %36 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%35, %weights : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded10 = tensor.expand_shape %bias [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
  %38 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%36, %expanded10 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %39 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%38 : tensor<8x112x32x32xbf16>)
    outs(%zero_init : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>

  return %39 : tensor<8x112x32x32xbf16>
}

