// RUN: fhe-tool -unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
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
// CHECK:     %0 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %2 = fhe.sub(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %3 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %4 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %5 = fhe.sub(%3, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %6 = fhe.multiply(%2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %7 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %8 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %9 = fhe.sub(%7, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %10 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %11 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %12 = fhe.sub(%10, %11) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %13 = fhe.multiply(%9, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %14 = fhe.add(%6, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %15 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %16 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %17 = fhe.sub(%15, %16) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %18 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %19 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %20 = fhe.sub(%18, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %21 = fhe.multiply(%17, %20) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %22 = fhe.add(%14, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %23 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %24 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %25 = fhe.sub(%23, %24) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %26 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %27 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %28 = fhe.sub(%26, %27) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %29 = fhe.multiply(%25, %28) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %30 = fhe.add(%22, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     return %30 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }