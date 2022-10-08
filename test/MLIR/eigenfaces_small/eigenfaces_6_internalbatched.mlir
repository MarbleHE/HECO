module {
  func.func private @encryptedEigenfaces(%arg0: !fhe.batched_secret<4 x f64>, %arg1: !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = linalg.init_tensor [4] : tensor<4x!fhe.secret<f64>>
    %1 = fhe.multiply(%arg1, %arg1) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %2 = fhe.multiply(%arg0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %3 = fhe.sub(%1, %2) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %4 = fhe.rotate(%2) by 3 : <4 x f64>
    %5 = fhe.sub(%1, %4) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %6 = fhe.rotate(%2) by 6 : <4 x f64>
    %7 = fhe.sub(%1, %6) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %8 = fhe.rotate(%2) by 9 : <4 x f64>
    %9 = fhe.sub(%1, %8) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %10 = fhe.add(%9, %7, %3, %5) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %11 = fhe.materialize(%0) : (tensor<4x!fhe.secret<f64>>) -> !fhe.batched_secret<4 x f64>
    %12 = fhe.combine(%10[0], %11) : !fhe.batched_secret<4 x f64>
    %13 = fhe.rotate(%1) by 15 : <16 x f64>
    %14 = fhe.sub(%13, %2) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %15 = fhe.rotate(%2) by 4 : <4 x f64>
    %16 = fhe.sub(%1, %15) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %17 = fhe.rotate(%2) by 7 : <4 x f64>
    %18 = fhe.sub(%1, %17) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %19 = fhe.rotate(%2) by 10 : <4 x f64>
    %20 = fhe.sub(%1, %19) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %21 = fhe.rotate(%14) by 1 : <4 x f64>
    %22 = fhe.add(%20, %18, %21, %16) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %23 = fhe.combine(%22[1], %12) : !fhe.batched_secret<4 x f64>
    %24 = fhe.rotate(%1) by 14 : <16 x f64>
    %25 = fhe.sub(%24, %2) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %26 = fhe.rotate(%2) by 5 : <4 x f64>
    %27 = fhe.sub(%1, %26) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %28 = fhe.rotate(%2) by 8 : <4 x f64>
    %29 = fhe.sub(%1, %28) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %30 = fhe.rotate(%2) by 11 : <4 x f64>
    %31 = fhe.sub(%1, %30) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %32 = fhe.rotate(%25) by 2 : <4 x f64>
    %33 = fhe.add(%31, %29, %32, %27) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %34 = fhe.combine(%33[2], %23) : !fhe.batched_secret<4 x f64>
    %35 = fhe.rotate(%1) by 13 : <16 x f64>
    %36 = fhe.sub(%35, %2) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %37 = fhe.rotate(%2) by 12 : <4 x f64>
    %38 = fhe.sub(%1, %37) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %39 = fhe.rotate(%36) by 3 : <4 x f64>
    %40 = fhe.add(%38, %9, %39, %7) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %41 = fhe.combine(%40[3], %34) : !fhe.batched_secret<4 x f64>
    return %41 : !fhe.batched_secret<4 x f64>
  }
}

