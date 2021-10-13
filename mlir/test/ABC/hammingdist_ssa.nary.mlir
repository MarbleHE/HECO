builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %0 = tensor.extract %arg0[%c0] : tensor<4xindex>
    %1 = tensor.extract %arg1[%c0] : tensor<4xindex>
    %2 = abc.fhe_sub(%0, %1) : (index, index) -> index
    %3 = tensor.extract %arg0[%c0] : tensor<4xindex>
    %4 = tensor.extract %arg1[%c0] : tensor<4xindex>
    %5 = abc.fhe_sub(%3, %4) : (index, index) -> index
    %6 = abc.fhe_mul(%2, %5) : (index, index) -> index
    %7 = tensor.extract %arg0[%c1] : tensor<4xindex>
    %8 = tensor.extract %arg1[%c1] : tensor<4xindex>
    %9 = abc.fhe_sub(%7, %8) : (index, index) -> index
    %10 = tensor.extract %arg0[%c1] : tensor<4xindex>
    %11 = tensor.extract %arg1[%c1] : tensor<4xindex>
    %12 = abc.fhe_sub(%10, %11) : (index, index) -> index
    %13 = abc.fhe_mul(%9, %12) : (index, index) -> index
    %14 = tensor.extract %arg0[%c2] : tensor<4xindex>
    %15 = tensor.extract %arg1[%c2] : tensor<4xindex>
    %16 = abc.fhe_sub(%14, %15) : (index, index) -> index
    %17 = tensor.extract %arg0[%c2] : tensor<4xindex>
    %18 = tensor.extract %arg1[%c2] : tensor<4xindex>
    %19 = abc.fhe_sub(%17, %18) : (index, index) -> index
    %20 = abc.fhe_mul(%16, %19) : (index, index) -> index
    %21 = tensor.extract %arg0[%c3] : tensor<4xindex>
    %22 = tensor.extract %arg1[%c3] : tensor<4xindex>
    %23 = abc.fhe_sub(%21, %22) : (index, index) -> index
    %24 = tensor.extract %arg0[%c3] : tensor<4xindex>
    %25 = tensor.extract %arg1[%c3] : tensor<4xindex>
    %26 = abc.fhe_sub(%24, %25) : (index, index) -> index
    %27 = abc.fhe_mul(%23, %26) : (index, index) -> index
    %28 = abc.fhe_add(%27, %20, %6, %13) : (index, index, index, index) -> index
    return %28 : index
  }
}

