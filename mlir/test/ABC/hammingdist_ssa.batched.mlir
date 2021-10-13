builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c3 = constant 3 : index
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = abc.fhe_rotate %arg0, %c0 : tensor<4xindex>, index, tensor<4xindex>
    %1 = abc.fhe_rotate %arg1, %c0 : tensor<4xindex>, index, tensor<4xindex>
    %2 = abc.fhe_sub(%0, %1) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %3 = abc.fhe_mul(%2, %2) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %4 = abc.fhe_rotate %arg0, %c1 : tensor<4xindex>, index, tensor<4xindex>
    %5 = abc.fhe_rotate %arg1, %c1 : tensor<4xindex>, index, tensor<4xindex>
    %6 = abc.fhe_sub(%4, %5) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %7 = abc.fhe_mul(%6, %6) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %8 = abc.fhe_rotate %arg0, %c2 : tensor<4xindex>, index, tensor<4xindex>
    %9 = abc.fhe_rotate %arg1, %c2 : tensor<4xindex>, index, tensor<4xindex>
    %10 = abc.fhe_sub(%8, %9) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %11 = abc.fhe_mul(%10, %10) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %12 = abc.fhe_rotate %arg0, %c3 : tensor<4xindex>, index, tensor<4xindex>
    %13 = abc.fhe_rotate %arg1, %c3 : tensor<4xindex>, index, tensor<4xindex>
    %14 = abc.fhe_sub(%12, %13) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %15 = abc.fhe_mul(%14, %14) : (tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    %16 = abc.fhe_add(%15, %11, %3, %7) : (tensor<4xindex>, tensor<4xindex>, tensor<4xindex>, tensor<4xindex>) -> tensor<4xindex>
    return %16 : tensor<4xindex>
  }
}

