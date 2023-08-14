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
    %1 = fhe.multiply(%0, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %2 = tensor.extract %arg0[%c1] : tensor<16xf64>
    %3 = fhe.multiply(%2, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %4 = tensor.extract %arg0[%c2] : tensor<16xf64>
    %5 = fhe.multiply(%4, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %6 = tensor.extract %arg0[%c3] : tensor<16xf64>
    %7 = fhe.multiply(%6, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    
    %8 = fhe.rotate(%7) by 1 : <4 x f64>
    %9 = fhe.rotate(%5) by 2 : <4 x f64>
    %10 = fhe.rotate(%3) by 3 : <4 x f64>
    %11 = fhe.add(%8, %9, %1, %10) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %12 = fhe.combine(%11[0], %arg1) : !fhe.batched_secret<4 x f64>
    

    %13 = tensor.extract %arg0[%c4] : tensor<16xf64>
    %14 = fhe.multiply(%13, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %15 = tensor.extract %arg0[%c5] : tensor<16xf64>
    %16 = fhe.multiply(%15, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %17 = tensor.extract %arg0[%c6] : tensor<16xf64>
    %18 = fhe.multiply(%17, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %19 = tensor.extract %arg0[%c7] : tensor<16xf64>
    %20 = fhe.multiply(%19, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    
    %21 = fhe.rotate(%20) by 2 : <4 x f64>
    %22 = fhe.rotate(%18) by 3 : <4 x f64>
    %23 = fhe.rotate(%14) by 1 : <4 x f64>
    %24 = fhe.add(%21, %22, %23, %16) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %25 = fhe.combine(%24[1], %12) : !fhe.batched_secret<4 x f64>
    
    
    %26 = tensor.extract %arg0[%c8] : tensor<16xf64>
    %27 = fhe.multiply(%26, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %28 = tensor.extract %arg0[%c9] : tensor<16xf64>
    %29 = fhe.multiply(%28, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %30 = tensor.extract %arg0[%c10] : tensor<16xf64>
    %31 = fhe.multiply(%30, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %32 = tensor.extract %arg0[%c11] : tensor<16xf64>
    %33 = fhe.multiply(%32, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
   
    %34 = fhe.rotate(%33) by 3 : <4 x f64>
    %35 = fhe.rotate(%27) by 2 : <4 x f64>
    %36 = fhe.rotate(%29) by 1 : <4 x f64>
    %37 = fhe.add(%34, %31, %35, %36) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %38 = fhe.combine(%37[2], %25) : !fhe.batched_secret<4 x f64>
    
    
    %39 = tensor.extract %arg0[%c12] : tensor<16xf64>
    %40 = fhe.multiply(%39, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %41 = tensor.extract %arg0[%c13] : tensor<16xf64>
    %42 = fhe.multiply(%41, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %43 = tensor.extract %arg0[%c14] : tensor<16xf64>
    %44 = fhe.multiply(%43, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %45 = tensor.extract %arg0[%c15] : tensor<16xf64>
    %46 = fhe.multiply(%45, %arg1) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    
    %47 = fhe.rotate(%44) by 1 : <4 x f64>
    %48 = fhe.rotate(%40) by 3 : <4 x f64>
    %49 = fhe.rotate(%42) by 2 : <4 x f64>
    %50 = fhe.add(%46, %47, %48, %49) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %51 = fhe.combine(%50[3], %38) : !fhe.batched_secret<4 x f64>
    
    return %51 : !fhe.batched_secret<4 x f64>
  }
}

