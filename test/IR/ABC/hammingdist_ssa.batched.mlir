builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> tensor<4xindex> {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fhe.rotate %arg0, %c0 : tensor<4xindex>, index, tensor<4xindex>
    %1 = fhe.rotate %arg1, %c0 : tensor<4xindex>, index, tensor<4xindex>
    %2 = fhe.sub(%0, %1) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %3 = fhe.mul(%2, %2) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %4 = fhe.rotate %arg0, %c1 : tensor<4xindex>, index, tensor<4xindex>
    %5 = fhe.rotate %arg1, %c1 : tensor<4xindex>, index, tensor<4xindex>
    %6 = fhe.sub(%4, %5) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %7 = fhe.mul(%6, %6) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %8 = fhe.rotate %arg0, %c2 : tensor<4xindex>, index, tensor<4xindex>
    %9 = fhe.rotate %arg1, %c2 : tensor<4xindex>, index, tensor<4xindex>
    %10 = fhe.sub(%8, %9) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %11 = fhe.mul(%10, %10) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %12 = fhe.rotate %arg0, %c3 : tensor<4xindex>, index, tensor<4xindex>
    %13 = fhe.rotate %arg1, %c3 : tensor<4xindex>, index, tensor<4xindex>
    %14 = fhe.sub(%12, %13) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %15 = fhe.mul(%14, %14) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %16 = fhe.add(%15, %11, %3, %7) : (tensor<4xindex>, tensor<4xindex>, tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    return %16 : tensor<4xindex>
  }
}

