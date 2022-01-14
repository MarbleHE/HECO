// RUN: abc-opt -unroll-loops --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = fhe.constant 0.000000e+00 : f64
    %1:4 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %arg1, %arg4 = %c64, %arg5 = %arg0, %arg6 = %0) -> (tensor<4x!fhe.secret<f64>>, index, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>) {
      %2 = tensor.extract %arg5[%arg2] : tensor<4x!fhe.secret<f64>>
      %3 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
      %4 = fhe.sub(%2, %3) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %5 = tensor.extract %arg5[%arg2] : tensor<4x!fhe.secret<f64>>
      %6 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
      %7 = fhe.sub(%5, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %8 = fhe.multiply(%4, %7) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %9 = fhe.add(%arg6, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      affine.yield %arg3, %arg4, %arg5, %9 : tensor<4x!fhe.secret<f64>>, index, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>
    }
    return %1#3 : !fhe.secret<f64>
  }
}


// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %0 = fhe.cast(%c0) : (index) -> !fhe.secret<f64>
// CHECK:     %1 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %2 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %3 = fhe.sub(%1, %2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %4 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %5 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %6 = fhe.sub(%4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %7 = fhe.multiply(%3, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %8 = fhe.add(%0, %7) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %9 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %10 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %11 = fhe.sub(%9, %10) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %12 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %13 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %14 = fhe.sub(%12, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %15 = fhe.multiply(%11, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %16 = fhe.add(%8, %15) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %17 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %18 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %19 = fhe.sub(%17, %18) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %20 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %21 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %22 = fhe.sub(%20, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %23 = fhe.multiply(%19, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %24 = fhe.add(%16, %23) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %25 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %26 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %27 = fhe.sub(%25, %26) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %28 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %29 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %30 = fhe.sub(%28, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %31 = fhe.multiply(%27, %30) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %32 = fhe.add(%24, %31) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     return %32 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }