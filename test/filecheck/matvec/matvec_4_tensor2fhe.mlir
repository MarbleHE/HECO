module {
  func.func private @encryptedMVP(%arg0: tensor<16xf64>, %arg1: !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64> {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c11 = arith.constant 11 : index
    %c1 = arith.constant 1 : index
    %c13 = arith.constant 13 : index
    %c2 = arith.constant 2 : index
    %c14 = arith.constant 14 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c15 = arith.constant 15 : index
    %0 = tensor.extract %arg0[%c0] : tensor<16xf64>
    %1 = fhe.extract %arg1[0] : <4 x f64>
    %2 = fhe.multiply(%0, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %3 = tensor.extract %arg0[%c1] : tensor<16xf64>
    %4 = fhe.extract %arg1[1] : <4 x f64>
    %5 = fhe.multiply(%3, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %6 = tensor.extract %arg0[%c2] : tensor<16xf64>
    %7 = fhe.extract %arg1[2] : <4 x f64>
    %8 = fhe.multiply(%6, %7) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %9 = tensor.extract %arg0[%c3] : tensor<16xf64>
    %10 = fhe.extract %arg1[3] : <4 x f64>
    %11 = fhe.multiply(%9, %10) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %12 = fhe.add(%11, %8, %2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %13 = fhe.insert %12 into %arg1[0] : <4 x f64>
    %14 = tensor.extract %arg0[%c4] : tensor<16xf64>
    %15 = fhe.multiply(%14, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %16 = tensor.extract %arg0[%c5] : tensor<16xf64>
    %17 = fhe.multiply(%16, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %18 = tensor.extract %arg0[%c6] : tensor<16xf64>
    %19 = fhe.multiply(%18, %7) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %20 = tensor.extract %arg0[%c7] : tensor<16xf64>
    %21 = fhe.multiply(%20, %10) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %22 = fhe.add(%21, %19, %15, %17) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %23 = fhe.insert %22 into %13[1] : <4 x f64>
    %24 = tensor.extract %arg0[%c8] : tensor<16xf64>
    %25 = fhe.multiply(%24, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %26 = tensor.extract %arg0[%c9] : tensor<16xf64>
    %27 = fhe.multiply(%26, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %28 = tensor.extract %arg0[%c10] : tensor<16xf64>
    %29 = fhe.multiply(%28, %7) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %30 = tensor.extract %arg0[%c11] : tensor<16xf64>
    %31 = fhe.multiply(%30, %10) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %32 = fhe.add(%31, %29, %25, %27) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %33 = fhe.insert %32 into %23[2] : <4 x f64>
    %34 = tensor.extract %arg0[%c12] : tensor<16xf64>
    %35 = fhe.multiply(%34, %1) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %36 = tensor.extract %arg0[%c13] : tensor<16xf64>
    %37 = fhe.multiply(%36, %4) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %38 = tensor.extract %arg0[%c14] : tensor<16xf64>
    %39 = fhe.multiply(%38, %7) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %40 = tensor.extract %arg0[%c15] : tensor<16xf64>
    %41 = fhe.multiply(%40, %10) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
    %42 = fhe.add(%41, %39, %35, %37) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %43 = fhe.insert %42 into %33[3] : <4 x f64>
    return %43 : !fhe.batched_secret<4 x f64>
  }
}

