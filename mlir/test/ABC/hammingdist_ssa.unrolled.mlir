builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c3 = constant 3 : index
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<4xindex>
    %1 = tensor.extract %arg1[%c0] : tensor<4xindex>
    %2 = subi %0, %1 : index
    %3 = muli %2, %2 : index
    %4 = tensor.extract %arg0[%c1] : tensor<4xindex>
    %5 = tensor.extract %arg1[%c1] : tensor<4xindex>
    %6 = subi %4, %5 : index
    %7 = muli %6, %6 : index
    %8 = addi %3, %7 : index
    %9 = tensor.extract %arg0[%c2] : tensor<4xindex>
    %10 = tensor.extract %arg1[%c2] : tensor<4xindex>
    %11 = subi %9, %10 : index
    %12 = muli %11, %11 : index
    %13 = addi %8, %12 : index
    %14 = tensor.extract %arg0[%c3] : tensor<4xindex>
    %15 = tensor.extract %arg1[%c3] : tensor<4xindex>
    %16 = subi %14, %15 : index
    %17 = muli %16, %16 : index
    %18 = addi %13, %17 : index
    return %18 : index
  }
}

