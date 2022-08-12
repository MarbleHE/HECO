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