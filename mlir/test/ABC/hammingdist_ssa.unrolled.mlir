builtin.module  {
  builtin.func private @encryptedHammingDistance(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> index {
    %c3 = constant 3 : index
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<4xindex>
    %1 = tensor.extract %arg1[%c0] : tensor<4xindex>
    %2 = subi %0, %1 : index
    %3 = tensor.extract %arg0[%c0] : tensor<4xindex>
    %4 = tensor.extract %arg1[%c0] : tensor<4xindex>
    %5 = subi %3, %4 : index
    %6 = muli %2, %5 : index
    %7 = tensor.extract %arg0[%c1] : tensor<4xindex>
    %8 = tensor.extract %arg1[%c1] : tensor<4xindex>
    %9 = subi %7, %8 : index
    %10 = tensor.extract %arg0[%c1] : tensor<4xindex>
    %11 = tensor.extract %arg1[%c1] : tensor<4xindex>
    %12 = subi %10, %11 : index
    %13 = muli %9, %12 : index
    %14 = addi %6, %13 : index
    %15 = tensor.extract %arg0[%c2] : tensor<4xindex>
    %16 = tensor.extract %arg1[%c2] : tensor<4xindex>
    %17 = subi %15, %16 : index
    %18 = tensor.extract %arg0[%c2] : tensor<4xindex>
    %19 = tensor.extract %arg1[%c2] : tensor<4xindex>
    %20 = subi %18, %19 : index
    %21 = muli %17, %20 : index
    %22 = addi %14, %21 : index
    %23 = tensor.extract %arg0[%c3] : tensor<4xindex>
    %24 = tensor.extract %arg1[%c3] : tensor<4xindex>
    %25 = subi %23, %24 : index
    %26 = tensor.extract %arg0[%c3] : tensor<4xindex>
    %27 = tensor.extract %arg1[%c3] : tensor<4xindex>
    %28 = subi %26, %27 : index
    %29 = muli %25, %28 : index
    %30 = addi %22, %29 : index
    return %30 : index
  }
}

