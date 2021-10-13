builtin.module  {
  builtin.func private @encryptedBoxBlur(%arg0: tensor<64xindex>) -> tensor<64xindex> {
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %c8 = constant 8 : index
    %0:2 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %arg0, %arg3 = %arg0) -> (tensor<64xindex>, tensor<64xindex>) {
      %1:3 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %arg2, %arg6 = %arg3, %arg7 = %arg1) -> (tensor<64xindex>, tensor<64xindex>, index) {
        %2:4 = affine.for %arg8 = -1 to 2 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %c0, %arg12 = %arg7) -> (tensor<64xindex>, tensor<64xindex>, index, index) {
          %6:4 = affine.for %arg13 = -1 to 2 iter_args(%arg14 = %arg9, %arg15 = %arg10, %arg16 = %arg11, %arg17 = %arg12) -> (tensor<64xindex>, tensor<64xindex>, index, index) {
            %7 = addi %arg17, %arg13 : index
            %8 = muli %7, %c8 : index
            %9 = addi %arg4, %arg8 : index
            %10 = addi %8, %9 : index
            %11 = remi_unsigned %10, %c64 : index
            %12 = tensor.extract %arg14[%11] : tensor<64xindex>
            %13 = addi %arg16, %12 : index
            affine.yield %arg14, %arg15, %13, %arg17 : tensor<64xindex>, tensor<64xindex>, index, index
          }
          affine.yield %6#0, %6#1, %6#2, %6#3 : tensor<64xindex>, tensor<64xindex>, index, index
        }
        %3 = muli %2#3, %c8 : index
        %4 = addi %3, %arg4 : index
        %5 = tensor.insert %2#2 into %2#1[%4] : tensor<64xindex>
        affine.yield %2#0, %5, %2#3 : tensor<64xindex>, tensor<64xindex>, index
      }
      affine.yield %1#0, %1#1 : tensor<64xindex>, tensor<64xindex>
    }
    return %0#1 : tensor<64xindex>
  }
}