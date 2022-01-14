module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = fhe.cast(%c0) : (index) -> !fhe.secret<f64>
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