#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @simple_copy(%I: tensor<6x9xf32>, %O: tensor<6x9xf32>) -> tensor<6x9xf32> {
  %OO = linalg.generic {indexing_maps = [#map0, #map1],
                        iterator_types = ["parallel", "parallel"]}
    ins(%I: tensor<6x9xf32>) outs(%O: tensor<6x9xf32>) {
      ^bb0(%i: f32, %o:f32):
        linalg.yield %i: f32
    } -> tensor<6x9xf32>
  return %OO: tensor<6x9xf32>
}
