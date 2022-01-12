builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> tensor<4xindex> {
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %2 = fhe.sub(%arg0, %arg1) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %3 = fhe.mul(%2, %2) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %4 = fhe.rotate %3, %c2 : tensor<4xindex>, index, tensor<4xindex>
    %5 = fhe.add(%4,%3) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %6 = fhe.rotate %5, %c1 : tensor<4xindex>, index, tensor<4xindex>
    %7 = fhe.add(%6,%5) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    return %7 : tensor<4xindex>
  }
}

