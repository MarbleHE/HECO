module {
  func.func private @encryptedMVP(%arg0: tensor<16xf64>, %arg1: tensor<4x!fhe.secret<f64>>) -> tensor<4x!fhe.secret<f64>> {
    %c15 = arith.constant 15 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c14 = arith.constant 14 : index
    %c2 = arith.constant 2 : index
    %c13 = arith.constant 13 : index
    %c1 = arith.constant 1 : index
    %c11 = arith.constant 11 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.extract %arg0[%c0] : tensor<16xf64>
    %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %2 = fhe.multiply(%0, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %3 = tensor.extract %arg0[%c1] : tensor<16xf64>
    %4 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %5 = fhe.multiply(%3, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %6 = fhe.add(%2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %7 = tensor.extract %arg0[%c2] : tensor<16xf64>
    %8 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    %9 = fhe.multiply(%7, %8) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %10 = fhe.add(%6, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %11 = tensor.extract %arg0[%c3] : tensor<16xf64>
    %12 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %13 = fhe.multiply(%11, %12) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %14 = fhe.add(%10, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %15 = tensor.insert %14 into %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %16 = tensor.extract %arg0[%c4] : tensor<16xf64>
    %17 = fhe.multiply(%16, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %18 = tensor.extract %arg0[%c5] : tensor<16xf64>
    %19 = fhe.multiply(%18, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %20 = fhe.add(%17, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %21 = tensor.extract %arg0[%c6] : tensor<16xf64>
    %22 = fhe.multiply(%21, %8) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %23 = fhe.add(%20, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %24 = tensor.extract %arg0[%c7] : tensor<16xf64>
    %25 = fhe.multiply(%24, %12) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %26 = fhe.add(%23, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %27 = tensor.insert %26 into %15[%c1] : tensor<4x!fhe.secret<f64>>
    %28 = tensor.extract %arg0[%c8] : tensor<16xf64>
    %29 = fhe.multiply(%28, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %30 = tensor.extract %arg0[%c9] : tensor<16xf64>
    %31 = fhe.multiply(%30, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %32 = fhe.add(%29, %31) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %33 = tensor.extract %arg0[%c10] : tensor<16xf64>
    %34 = fhe.multiply(%33, %8) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %35 = fhe.add(%32, %34) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %36 = tensor.extract %arg0[%c11] : tensor<16xf64>
    %37 = fhe.multiply(%36, %12) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %38 = fhe.add(%35, %37) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %39 = tensor.insert %38 into %27[%c2] : tensor<4x!fhe.secret<f64>>
    %40 = tensor.extract %arg0[%c12] : tensor<16xf64>
    %41 = fhe.multiply(%40, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %42 = tensor.extract %arg0[%c13] : tensor<16xf64>
    %43 = fhe.multiply(%42, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %44 = fhe.add(%41, %43) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %45 = tensor.extract %arg0[%c14] : tensor<16xf64>
    %46 = fhe.multiply(%45, %8) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %47 = fhe.add(%44, %46) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %48 = tensor.extract %arg0[%c15] : tensor<16xf64>
    %49 = fhe.multiply(%48, %12) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %50 = fhe.add(%47, %49) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %51 = tensor.insert %50 into %39[%c3] : tensor<4x!fhe.secret<f64>>
    return %51 : tensor<4x!fhe.secret<f64>>
  }
}

