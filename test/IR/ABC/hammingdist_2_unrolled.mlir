builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %2 = fhe.sub %0, %1 : !fhe.secret<f64>
    %3 = fhe.mul %2, %2 : !fhe.secret<f64>
    %4 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %5 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %6 = fhe.sub %4, %5 : !fhe.secret<f64>
    %7 = fhe.mul %6, %6 : !fhe.secret<f64>
    %8 = fhe.add %3, %7 : !fhe.secret<f64>
    %9 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %10 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %11 = fhe.sub %9, %10 : !fhe.secret<f64>
    %12 = fhe.mul %11, %11 : !fhe.secret<f64>
    %13 = fhe.add %8, %12 : !fhe.secret<f64>
    %14 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %15 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %16 = fhe.sub %14, %15 : !fhe.secret<f64>
    %17 = fhe.mul %16, %16 : !fhe.secret<f64>
    %18 = fhe.add %13, %17 : !fhe.secret<f64>
    return %18 : !fhe.secret<f64>
  }
}

