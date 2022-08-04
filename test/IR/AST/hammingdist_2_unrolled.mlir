// RUN: abc-opt -nary --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %2 = fhe.sub(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %3 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %4 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %5 = fhe.sub(%3, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %6 = fhe.multiply(%2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %7 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %8 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %9 = fhe.sub(%7, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %10 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %11 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %12 = fhe.sub(%10, %11) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %13 = fhe.multiply(%9, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %14 = fhe.add(%6, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %15 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %16 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %17 = fhe.sub(%15, %16) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %18 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %19 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %20 = fhe.sub(%18, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %21 = fhe.multiply(%17, %20) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %22 = fhe.add(%14, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %23 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %24 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %25 = fhe.sub(%23, %24) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %26 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %27 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %28 = fhe.sub(%26, %27) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %29 = fhe.multiply(%25, %28) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %30 = fhe.add(%22, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    return %30 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c3 = arith.constant 3 : index
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
// CHECK:     %14 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %15 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %16 = fhe.sub(%14, %15) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %17 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %18 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %19 = fhe.sub(%17, %18) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %20 = fhe.multiply(%16, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %21 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %22 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %23 = fhe.sub(%21, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %24 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %25 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %26 = fhe.sub(%24, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %27 = fhe.multiply(%23, %26) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %28 = fhe.add(%27, %20, %6, %13) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     return %28 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }