// RUN: abc-opt -nary --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_sf64 = fhe.constant 0.000000e+00 : f64
    %0 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %2 = fhe.sub(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %3 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %4 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %5 = fhe.sub(%3, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %6 = fhe.multiply(%2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %7 = fhe.add(%6, %c0_sf64) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %8 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %9 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %10 = fhe.sub(%8, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %11 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %12 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %13 = fhe.sub(%11, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %14 = fhe.multiply(%10, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %15 = fhe.add(%7, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %16 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %17 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %18 = fhe.sub(%16, %17) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %19 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %20 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %21 = fhe.sub(%19, %20) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %22 = fhe.multiply(%18, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %23 = fhe.add(%15, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %24 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %25 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %26 = fhe.sub(%24, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %27 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %28 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %29 = fhe.sub(%27, %28) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %30 = fhe.multiply(%26, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %31 = fhe.add(%23, %30) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    return %31 : !fhe.secret<f64>
  }
}