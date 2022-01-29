builtin.module  {
  builtin.func private @encryptedBoxBlur(%arg0: tensor<64xindex>) -> tensor<64xindex> {
    %c8 = constant 8 : index
    %c64 = constant 64 : index
    %unused, %used = affine.for %arg1 = 0 to 8  iter_args(%foo = %c8, %boo = %c64) -> (index, index) {
      %new_boo = std.addi %boo, %arg1 : index
      affine.yield %foo, %new_boo : index, index
    }
    %updated = tensor.insert %c64 into %arg0[%c8] : tensor<64xindex>
    return %updated : tensor<64xindex>
  }
}

// INSIGHT:

