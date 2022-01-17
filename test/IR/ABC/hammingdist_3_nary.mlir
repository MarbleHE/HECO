//RUN:  abc-opt -tensor2fhe --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
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
    %14 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %15 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %16 = fhe.sub(%14, %15) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %17 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %18 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %19 = fhe.sub(%17, %18) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %20 = fhe.multiply(%16, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %21 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %22 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %23 = fhe.sub(%21, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %24 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %25 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %26 = fhe.sub(%24, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %27 = fhe.multiply(%23, %26) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %28 = fhe.add(%27, %20, %6, %13) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    return %28 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
// CHECK:     %0 = fhe.extract %arg0[0] : <f64>
// CHECK:     %1 = fhe.extract %arg1[0] : <f64>
// CHECK:     %2 = fhe.sub(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %3 = fhe.multiply(%2, %2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %4 = fhe.extract %arg0[1] : <f64>
// CHECK:     %5 = fhe.extract %arg1[1] : <f64>
// CHECK:     %6 = fhe.sub(%4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %7 = fhe.multiply(%6, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %8 = fhe.extract %arg0[2] : <f64>
// CHECK:     %9 = fhe.extract %arg1[2] : <f64>
// CHECK:     %10 = fhe.sub(%8, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %11 = fhe.multiply(%10, %10) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %12 = fhe.extract %arg0[3] : <f64>
// CHECK:     %13 = fhe.extract %arg1[3] : <f64>
// CHECK:     %14 = fhe.sub(%12, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %15 = fhe.multiply(%14, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %16 = fhe.add(%15, %11, %3, %7) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     return %16 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }