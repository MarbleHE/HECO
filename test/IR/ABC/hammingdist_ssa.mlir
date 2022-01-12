builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c0 = arith.constant 0 : index
    %0:3 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %arg0, %arg4 = %arg1, %arg5 = %c0) -> (tensor<4x!fhe.secret<f64>>, tensor<4x!fhe.secret<f64>>, f64) {
      %1 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
      %2 = tensor.extract %arg4[%arg2] : tensor<4x!fhe.secret<f64>>
      %3 = fhe.sub %1, %2 : !fhe.secret<f64>
      %4 = fhe.mul %3, %3 : !fhe.secret<f64>
      %5 = fhe.add %arg5, %4 : !fhe.secret<f64>
      affine.yield %arg3, %arg4, %5 : tensor<4x!fhe.secret<f64>>, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>
    }
    return %0#2 : index
  }
}
