builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.extract %arg0[%c0] : tensor<4xindex>
    %1 = tensor.extract %arg1[%c0] : tensor<4xindex>
    %2 = fhe.sub(%0, %1) : (index, index) -> index
    %3 = fhe.mul(%2, %2) : (index, index) -> index
    %4 = tensor.extract %arg0[%c1] : tensor<4xindex>
    %5 = tensor.extract %arg1[%c1] : tensor<4xindex>
    %6 = fhe.sub(%4, %5) : (index, index) -> index
    %7 = fhe.mul(%6, %6) : (index, index) -> index
    %8 = tensor.extract %arg0[%c2] : tensor<4xindex>
    %9 = tensor.extract %arg1[%c2] : tensor<4xindex>
    %10 = fhe.sub(%8, %9) : (index, index) -> index
    %11 = fhe.mul(%10, %10) : (index, index) -> index
    %12 = tensor.extract %arg0[%c3] : tensor<4xindex>
    %13 = tensor.extract %arg1[%c3] : tensor<4xindex>
    %14 = fhe.sub(%12, %13) : (index, index) -> index
    %15 = fhe.mul(%14, %14) : (index, index) -> index
    %16 = fhe.add(%15, %11, %3, %7) : (index, index, index, index) -> index
    return %16 : index
  }
}

