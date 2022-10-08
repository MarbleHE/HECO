module {
  func.func private @encryptedEigenfaces(%arg0: !fhe.batched_secret<4 x f64>, %arg1: !fhe.batched_secret<16 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = linalg.init_tensor [4] : tensor<4x!fhe.secret<f64>>
    %22 = fhe.materialize(%0) : (tensor<4x!fhe.secret<f64>>) -> !fhe.batched_secret<4 x f64>

    // SQUARE(x)
    %3 = fhe.extract %arg0[0] : <4 x f64>
    %4 = fhe.multiply(%3, %3) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %8 = fhe.extract %arg0[1] : <4 x f64>
    %9 = fhe.multiply(%8, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %13 = fhe.extract %arg0[2] : <4 x f64>
    %14 = fhe.multiply(%13, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %18 = fhe.extract %arg0[3] : <4 x f64>
    %19 = fhe.multiply(%18, %18) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
  
    // FIRST ELEMENT
    %1 = fhe.extract %arg1[0] : <16 x f64>
    %2 = fhe.multiply(%1, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>    
    %5 = fhe.sub(%2, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %6 = fhe.extract %arg1[1] : <16 x f64>
    %7 = fhe.multiply(%6, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %10 = fhe.sub(%7, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %11 = fhe.extract %arg1[2] : <16 x f64>
    %12 = fhe.multiply(%11, %11) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>    
    %15 = fhe.sub(%12, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %16 = fhe.extract %arg1[3] : <16 x f64>
    %17 = fhe.multiply(%16, %16) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %20 = fhe.sub(%17, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    %21 = fhe.add(%20, %15, %5, %10) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %23 = fhe.insert %21 into %22[0] : <4 x f64>

    // SECOND ELEMENT
    %24 = fhe.extract %arg1[4] : <16 x f64>
    %25 = fhe.multiply(%24, %24) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %26 = fhe.sub(%25, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %27 = fhe.extract %arg1[5] : <16 x f64>
    %28 = fhe.multiply(%27, %27) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %29 = fhe.sub(%28, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %30 = fhe.extract %arg1[6] : <16 x f64>
    %31 = fhe.multiply(%30, %30) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %32 = fhe.sub(%31, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %33 = fhe.extract %arg1[7] : <16 x f64>
    %34 = fhe.multiply(%33, %33) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %35 = fhe.sub(%34, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    %36 = fhe.add(%35, %32, %26, %29) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %37 = fhe.insert %36 into %23[1] : <4 x f64>
    
    // THIRD ELEMENT
    %38 = fhe.extract %arg1[8] : <16 x f64>
    %39 = fhe.multiply(%38, %38) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %40 = fhe.sub(%39, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %41 = fhe.extract %arg1[9] : <16 x f64>
    %42 = fhe.multiply(%41, %41) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %43 = fhe.sub(%42, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %44 = fhe.extract %arg1[10] : <16 x f64>
    %45 = fhe.multiply(%44, %44) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %46 = fhe.sub(%45, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %47 = fhe.extract %arg1[11] : <16 x f64>
    %48 = fhe.multiply(%47, %47) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %49 = fhe.sub(%48, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    %50 = fhe.add(%49, %46, %40, %43) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %51 = fhe.insert %50 into %37[2] : <4 x f64>
    
    // FOURTH ELEMENT
    %52 = fhe.extract %arg1[12] : <16 x f64>
    %53 = fhe.multiply(%52, %52) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %54 = fhe.sub(%53, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %55 = fhe.extract %arg1[13] : <16 x f64>
    %56 = fhe.multiply(%55, %55) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %57 = fhe.sub(%56, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %58 = fhe.extract %arg1[14] : <16 x f64>
    %59 = fhe.multiply(%58, %58) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %60 = fhe.sub(%59, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %61 = fhe.extract %arg1[15] : <16 x f64>
    %62 = fhe.multiply(%61, %61) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %63 = fhe.sub(%62, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    %64 = fhe.add(%63, %60, %54, %57) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %65 = fhe.insert %64 into %51[3] : <4 x f64>
    
    return %65 : !fhe.batched_secret<4 x f64>
  }
}

