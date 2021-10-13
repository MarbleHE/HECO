builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c0 = constant 0 : index
    %0:3 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %arg0, %arg4 = %arg1, %arg5 = %c0) -> (tensor<4xindex>, tensor<4xindex>, index) {
      %1 = tensor.extract %arg3[%arg2] : tensor<4xindex>
      %2 = tensor.extract %arg4[%arg2] : tensor<4xindex>
      %3 = subi %1, %2 : index
      %4 = tensor.extract %arg3[%arg2] : tensor<4xindex>
      %5 = tensor.extract %arg4[%arg2] : tensor<4xindex>
      %6 = subi %4, %5 : index
      %7 = muli %3, %6 : index
      %8 = addi %arg5, %7 : index
      affine.yield %arg3, %arg4, %8 : tensor<4xindex>, tensor<4xindex>, index
    }
    return %0#2 : index
  }
}

