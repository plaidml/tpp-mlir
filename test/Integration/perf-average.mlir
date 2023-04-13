// RUN: tpp-run %s -e entry -entry-point-result=void -seed 123 -splat-to-random -init-type=random | \
// RUN: FileCheck %s

memref.global "private" constant @__data_random : memref<32xf64> = dense<1.000000e+00>
memref.global "private" constant @__data_odd_sorted : memref<5xf64> = dense<[1.0, 3.0, 5.0, 7.0, 15.0]>
memref.global "private" constant @__data_odd_unsorted : memref<5xf64> = dense<[5.0, 7.0, 15.0, 1.0, 3.0]>
memref.global "private" constant @__data_even_sorted : memref<6xf64> = dense<[1.0, 2.0, 3.0, 5.0, 8.0, 9.0]>
memref.global "private" constant @__data_even_unsorted : memref<6xf64> = dense<[8.0, 5.0, 1.0, 9.0, 3.0, 2.0]>

func.func private @printResult(%result : f64) {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.000000e+00 : f64
  %buf = memref.alloca() : memref<1xf64>
  memref.store %result, %buf[%c0] : memref<1xf64>

  %v0 = vector.transfer_read %buf[%c0], %d1 : memref<1xf64>, vector<1xf64>
  vector.print %v0 : vector<1xf64>

  return
}

func.func private @runTest(%buf : memref<*xf64>) {
  %mean = perf.mean(%buf : memref<*xf64>) : f64
  call @printResult(%mean) : (f64) -> ()
  %median = perf.median(%buf : memref<*xf64>) : f64
  call @printResult(%median) : (f64) -> ()

  return
}

func.func @entry(%dummy: memref<1xf32>) {
  %0 = memref.get_global @__data_random : memref<32xf64>
  %b0 = memref.cast %0 : memref<32xf64> to memref<*xf64>
  call @runTest(%b0) : (memref<*xf64>) -> ()

  %1 = memref.get_global @__data_odd_sorted : memref<5xf64>
  %b1 = memref.cast %1 : memref<5xf64> to memref<*xf64>
  call @runTest(%b1) : (memref<*xf64>) -> ()

  %2 = memref.get_global @__data_odd_unsorted : memref<5xf64>
  %b2 = memref.cast %2 : memref<5xf64> to memref<*xf64>
  call @runTest(%b2) : (memref<*xf64>) -> ()

  %3 = memref.get_global @__data_even_sorted : memref<6xf64>
  %b3 = memref.cast %3 : memref<6xf64> to memref<*xf64>
  call @runTest(%b3) : (memref<*xf64>) -> ()

  %4 = memref.get_global @__data_even_unsorted : memref<6xf64>
  %b4 = memref.cast %4 : memref<6xf64> to memref<*xf64>
  call @runTest(%b4) : (memref<*xf64>) -> ()

  return
}

// CHECK: ( 0.481646 )
// CHECK: ( 0.443908 )
// CHECK: ( 6.2 )
// CHECK: ( 5 )
// CHECK: ( 6.2 )
// CHECK: ( 5 )
// CHECK: ( 4.66667 )
// CHECK: ( 4 )
// CHECK: ( 4.66667 )
// CHECK: ( 4 )
