module {
  func.func private @encryptedEigenfaces(%arg0: !fhe.batched_secret<4 x f64>, %arg1: !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = linalg.init_tensor [4] : tensor<4x!fhe.secret<f64>>
    %10 = fhe.materialize(%0) : (tensor<4x!fhe.secret<f64>>) -> !fhe.batched_secret<4 x f64>

    // SQUARE
    %1 = fhe.multiply(%arg1, %arg1) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %2 = fhe.multiply(%arg0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %3 = fhe.materialize(%2) : (!fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<16 x f64>

    // FIRST DB ELEMENT
    %4 = fhe.sub(%1, %3) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>

    %5 = fhe.rotate(%4) by 13 : <16 x f64>
    %6 = fhe.rotate(%4) by 14 : <16 x f64>
    %7 = fhe.rotate(%4) by 15 : <16 x f64>
    
    %8 = fhe.add(%5, %6, %4, %7) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %9 = fhe.extract %8[0] : <16 x f64>
    %11 = fhe.insert %9 into %10[0] : <4 x f64>

    // SECOND DB ELEMENT
    %12 = fhe.rotate(%3) by 4 : <16 x f64>
    %13 = fhe.sub(%1, %12) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    
    %14 = fhe.rotate(%13) by 10 : <16 x f64>
    %15 = fhe.rotate(%13) by 11 : <16 x f64>
    %16 = fhe.rotate(%13) by 13 : <16 x f64>
    %17 = fhe.rotate(%13) by 12 : <16 x f64>
    
    %18 = fhe.add(%14, %15, %16, %17) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %19 = fhe.extract %18[1] : <16 x f64>
    %20 = fhe.insert %19 into %11[1] : <4 x f64>

    // THIRD DB ELEMENT
    %21 = fhe.rotate(%3) by 8 : <16 x f64>
    %22 = fhe.sub(%1, %21) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    
    %23 = fhe.rotate(%22) by 7 : <16 x f64>
    %24 = fhe.rotate(%22) by 8 : <16 x f64>
    %25 = fhe.rotate(%22) by 10 : <16 x f64>
    %26 = fhe.rotate(%22) by 9 : <16 x f64>
    
    %27 = fhe.add(%23, %24, %25, %26) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %28 = fhe.extract %27[2] : <16 x f64>
    %29 = fhe.insert %28 into %20[2] : <4 x f64>

    // FOURTH DB ELEMENT
    %30 = fhe.rotate(%3) by 12 : <16 x f64>
    %31 = fhe.sub(%1, %30) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    
    %32 = fhe.rotate(%31) by 4 : <16 x f64>
    %33 = fhe.rotate(%31) by 5 : <16 x f64>
    %34 = fhe.rotate(%31) by 7 : <16 x f64>
    %35 = fhe.rotate(%31) by 6 : <16 x f64>
    
    %36 = fhe.add(%32, %33, %34, %35) : (!fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>, !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<16 x f64>
    %37 = fhe.extract %36[3] : <16 x f64>
    %38 = fhe.insert %37 into %29[3] : <4 x f64>

    return %38 : !fhe.batched_secret<4 x f64>
  }
}

