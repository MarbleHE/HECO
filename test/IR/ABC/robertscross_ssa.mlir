builtin.module  {
  builtin.func private @encryptedBoxBlur(%arg0: tensor<64xindex>) -> tensor<64xindex> {
    %c0 = constant 0 : index
    %c8 = constant 8 : index
    %0:2 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %arg0, %arg3 = %arg0) -> (tensor<64xindex>, tensor<64xindex>) {
      %1:3 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %arg2, %arg6 = %arg3, %arg7 = %arg1) -> (tensor<64xindex>, tensor<64xindex>, index) {
        %2 = muli %arg7, %c8 : index
        %3 = addi %2, %arg4 : index
        %4 = tensor.insert %c0 into %arg6[%3] : tensor<64xindex>
        affine.yield %arg5, %4, %arg7 : tensor<64xindex>, tensor<64xindex>, index
      }
      affine.yield %1#0, %1#1 : tensor<64xindex>, tensor<64xindex>
    }
    return %0#1 : tensor<64xindex>
  }
}

