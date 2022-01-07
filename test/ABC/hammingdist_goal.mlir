builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> tensor<4xindex> {
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %2 = abc.fhe_sub(%arg0, %arg1) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %3 = abc.fhe_mul(%2, %2) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %4 = abc.fhe_rotate %3, %c2 : tensor<4xindex>, index, tensor<4xindex>
    %5 = abc.fhe_add(%4,%3) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %6 = abc.fhe_rotate %5, %c1 : tensor<4xindex>, index, tensor<4xindex>
    %7 = abc.fhe_add(%6,%5) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    return %7 : tensor<4xindex>
  }
}

