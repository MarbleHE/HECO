module  {
  func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fhe.cast(%c0) : (index) -> !fhe.secret<f64>
    %1 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %2 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %3 = fhe.sub(%1, %2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %4 = tensor.extract %arg0[%c0] : tensor<4x!fhe.secret<f64>>
    %5 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %6 = fhe.sub(%4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %7 = fhe.multiply(%3, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %8 = fhe.add(%0, %7) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %9 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %10 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %11 = fhe.sub(%9, %10) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %12 = tensor.extract %arg0[%c1] : tensor<4x!fhe.secret<f64>>
    %13 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %14 = fhe.sub(%12, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %15 = fhe.multiply(%11, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %16 = fhe.add(%8, %15) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %17 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %18 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %19 = fhe.sub(%17, %18) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %20 = tensor.extract %arg0[%c2] : tensor<4x!fhe.secret<f64>>
    %21 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %22 = fhe.sub(%20, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %23 = fhe.multiply(%19, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %24 = fhe.add(%16, %23) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %25 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %26 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %27 = fhe.sub(%25, %26) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %28 = tensor.extract %arg0[%c3] : tensor<4x!fhe.secret<f64>>
    %29 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %30 = fhe.sub(%28, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %31 = fhe.multiply(%27, %30) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %32 = fhe.add(%24, %31) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    return %32 : !fhe.secret<f64>
  }
}