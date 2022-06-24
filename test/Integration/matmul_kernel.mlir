// TODO: echo $(pwd)
// TODO: export C_INCLUDE_PATH=/Users/lchelini/llvm-project-orig/build/lib/clang/15.0.0/include
// TODO: clang -v -O3 -emit-llvm -S /Users/lchelini/tpp-sandbox/test/Integration/matmul_c_driver.c 
// TODO: llc matmul_c_driver.ll
// TODO: standalone-opt %s -tpp-compiler | mlir-translate -mlir-to-llvmir -o matmul_kernel.ll
// TODO: llc matmul_kernel.ll
// TODO: clang -O3 matmul_c_driver.s matmul_kernel.s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%A: tensor<6x9xf32>, %B: tensor<9x12xf32>,
                  %C: tensor<6x12xf32>) -> tensor<6x12xf32> {
  %D = linalg.generic {indexing_maps = [#map0, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B: tensor<6x9xf32>, tensor<9x12xf32>) outs(%C: tensor<6x12xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %c, %0 : f32
        linalg.yield %1 : f32
    } -> tensor<6x12xf32>
  return %D : tensor<6x12xf32>
}
