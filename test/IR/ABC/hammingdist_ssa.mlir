builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c0 = arith.constant 0 : index
    %0:3 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %arg0, %arg4 = %arg1, %arg5 = %c0) -> (tensor<4xindex>, tensor<4xindex>, index) {
      %1 = tensor.extract %arg3[%arg2] : tensor<4xindex>
      %2 = tensor.extract %arg4[%arg2] : tensor<4xindex>
      %3 = arith.subi %1, %2 : index
      %4 = arith.muli %3, %3 : index
      %5 = arith.addi %arg5, %4 : index
      affine.yield %arg3, %arg4, %5 : tensor<4xindex>, tensor<4xindex>, index
    }
    return %0#2 : index
  }
}
