// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void

func.func @entry() {
 %a = arith.constant 1:i1
 check.expect_true(%a):i1
 %b = arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.4 ]
    ]> : tensor<4x4xf32>
 %c =  arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.35 ]
    ]> : tensor<4x4xf32>

 %threshold = arith.constant 0.1: f32
 check.expect_almost_eq(%b, %c, %threshold):tensor<4x4xf32>, tensor<4x4xf32>, f32

 check.expect_sane(%b):tensor<4x4xf32>

 return
}
