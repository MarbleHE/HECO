module {
  func.func private @encryptedEigenfaces(%arg0: !fhe.batched_secret<4 x f64>, %arg1: !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = linalg.init_tensor [4] : tensor<4x!fhe.secret<f64>>
    %1 = fhe.multiply(%arg1, %arg1) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %2 = fhe.multiply(%arg0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %3 = fhe.materialize(%2) : (!fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<16 x f64>
    %4 = fhe.sub(%1, %3) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %5 = fhe.rotate(%4) by 2 : <16 x f64>
    %6 = fhe.add(%4, %5) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %7 = fhe.rotate(%6) by 1 : <16 x f64>
    %8 = fhe.add(%6, %7) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %9 = fhe.rotate(%8) by 3 : <16 x f64>
    %10 = fhe.extract %9[0] : <16 x f64>
    %11 = fhe.materialize(%0) : (tensor<4x!fhe.secret<f64>>) -> !fhe.batched_secret<4 x f64>
    %12 = fhe.insert %10 into %11[0] : <4 x f64>
    %13 = fhe.rotate(%3) by 4 : <16 x f64>
    %14 = fhe.sub(%1, %13) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %15 = fhe.rotate(%14) by 2 : <16 x f64>
    %16 = fhe.add(%14, %15) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %17 = fhe.rotate(%16) by 1 : <16 x f64>
    %18 = fhe.add(%16, %17) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %19 = fhe.rotate(%18) by 1 : <16 x f64>
    %20 = fhe.extract %19[1] : <16 x f64>
    %21 = fhe.insert %20 into %12[1] : <4 x f64>
    %22 = fhe.rotate(%3) by 8 : <16 x f64>
    %23 = fhe.sub(%1, %22) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %24 = fhe.rotate(%23) by 2 : <16 x f64>
    %25 = fhe.add(%23, %24) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %26 = fhe.rotate(%25) by 1 : <16 x f64>
    %27 = fhe.add(%25, %26) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %28 = fhe.extract %27[2] : <16 x f64>
    %29 = fhe.insert %28 into %21[2] : <4 x f64>
    %30 = fhe.rotate(%3) by 12 : <16 x f64>
    %31 = fhe.sub(%1, %30) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %32 = fhe.rotate(%31) by 2 : <16 x f64>
    %33 = fhe.add(%31, %32) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %34 = fhe.rotate(%33) by 1 : <16 x f64>
    %35 = fhe.add(%33, %34) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %36 = fhe.rotate(%35) by 15 : <16 x f64>
    %37 = fhe.extract %36[3] : <16 x f64>
    %38 = fhe.insert %37 into %29[3] : <4 x f64>
    return %38 : !fhe.batched_secret<4 x f64>
  }
}

