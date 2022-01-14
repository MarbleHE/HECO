// RUN: abc-opt -unroll-loops --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c0_sf64 = fhe.constant 0.000000e+00 : f64
    %0:4 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %arg1, %arg4 = %c64, %arg5 = %arg0, %arg6 = %c0_sf64) -> (tensor<4x!fhe.secret<f64>>, index, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>) {
      %1 = tensor.extract %arg5[%arg2] : tensor<4x!fhe.secret<f64>>
      %2 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
      %3 = fhe.sub(%1, %2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %4 = tensor.extract %arg5[%arg2] : tensor<4x!fhe.secret<f64>>
      %5 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
      %6 = fhe.sub(%4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %7 = fhe.multiply(%3, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %8 = fhe.add(%arg6, %7) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      affine.yield %arg3, %arg4, %arg5, %8 : tensor<4x!fhe.secret<f64>>, index, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>
    }
    return %0#3 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c0_sf64 = fhe.constant 0.000000e+00 : f64
// CHECK:     %0 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %2 = fhe.sub(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %3 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %4 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %5 = fhe.sub(%3, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %6 = fhe.multiply(%2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %7 = fhe.add(%6, %c0_sf64) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %8 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %9 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %10 = fhe.sub(%8, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %11 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %12 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %13 = fhe.sub(%11, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %14 = fhe.multiply(%10, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %15 = fhe.add(%7, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %16 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %17 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %18 = fhe.sub(%16, %17) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %19 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %20 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %21 = fhe.sub(%19, %20) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %22 = fhe.multiply(%18, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %23 = fhe.add(%15, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %24 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %25 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %26 = fhe.sub(%24, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %27 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %28 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %29 = fhe.sub(%27, %28) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %30 = fhe.multiply(%26, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %31 = fhe.add(%23, %30) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     return %31 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }