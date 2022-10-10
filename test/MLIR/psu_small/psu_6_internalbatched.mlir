module {
  func.func private @encryptedPSU(%arg0: !fhe.batched_secret<8 x f64>, %arg1: !fhe.batched_secret<4 x f64>, %arg2: !fhe.batched_secret<8 x f64>, %arg3: !fhe.batched_secret<4 x f64>) -> !fhe.secret<f64> {
    %cst = fhe.constant dense<1.000000e+00> : tensor<1xf64>
    %0 = fhe.materialize(%arg1) : (!fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<8 x f64>
    %1 = fhe.sub(%arg0, %arg2) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %2 = fhe.multiply(%1, %1) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %3 = fhe.sub(%cst, %2) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %4 = fhe.rotate(%3) by 1 : <8 x f64>
    %5 = fhe.multiply(%3, %4) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %6 = fhe.sub(%cst, %5) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %7 = fhe.rotate(%arg2) by 6 : <8 x f64>
    %8 = fhe.sub(%arg0, %7) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %9 = fhe.multiply(%8, %8) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %10 = fhe.sub(%cst, %9) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %11 = fhe.rotate(%10) by 1 : <8 x f64>
    %12 = fhe.multiply(%10, %11) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %13 = fhe.sub(%cst, %12) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %14 = fhe.rotate(%arg2) by 4 : <8 x f64>
    %15 = fhe.sub(%arg0, %14) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %16 = fhe.multiply(%15, %15) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %17 = fhe.sub(%cst, %16) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %18 = fhe.rotate(%17) by 1 : <8 x f64>
    %19 = fhe.multiply(%17, %18) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %20 = fhe.sub(%cst, %19) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %21 = fhe.rotate(%arg2) by 2 : <8 x f64>
    %22 = fhe.sub(%arg0, %21) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %23 = fhe.multiply(%22, %22) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %24 = fhe.sub(%cst, %23) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %25 = fhe.rotate(%24) by 1 : <8 x f64>
    %26 = fhe.multiply(%24, %25) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %27 = fhe.sub(%cst, %26) : (!fhe.batched_secret<-24 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %28 = fhe.materialize(%arg3) : (!fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<8 x f64>
    %29 = fhe.rotate(%27) by 7 : <8 x f64>
    %30 = fhe.rotate(%20) by 7 : <8 x f64>
    %31 = fhe.rotate(%13) by 7 : <8 x f64>
    %32 = fhe.rotate(%6) by 7 : <8 x f64>
    %33 = fhe.multiply(%28, %29, %30, %31, %32) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %34 = fhe.rotate(%20) by 6 : <8 x f64>
    %35 = fhe.rotate(%13) by 6 : <8 x f64>
    %36 = fhe.rotate(%6) by 6 : <8 x f64>
    %37 = fhe.rotate(%27) by 6 : <8 x f64>
    %38 = fhe.multiply(%28, %34, %35, %36, %37) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %39 = fhe.rotate(%13) by 5 : <8 x f64>
    %40 = fhe.rotate(%6) by 5 : <8 x f64>
    %41 = fhe.rotate(%27) by 5 : <8 x f64>
    %42 = fhe.rotate(%20) by 5 : <8 x f64>
    %43 = fhe.multiply(%28, %39, %40, %41, %42) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %44 = fhe.rotate(%6) by 4 : <8 x f64>
    %45 = fhe.rotate(%27) by 4 : <8 x f64>
    %46 = fhe.rotate(%20) by 4 : <8 x f64>
    %47 = fhe.rotate(%13) by 4 : <8 x f64>
    %48 = fhe.multiply(%28, %44, %45, %46, %47) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %49 = fhe.rotate(%48) by 5 : <8 x f64>
    %50 = fhe.rotate(%43) by 6 : <8 x f64>
    %51 = fhe.rotate(%38) by 7 : <8 x f64>
    %52 = fhe.rotate(%0) by 2 : <8 x f64>
    %53 = fhe.add(%0, %52) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %54 = fhe.rotate(%53) by 1 : <8 x f64>
    %55 = fhe.add(%53, %54) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %56 = fhe.rotate(%55) by 3 : <8 x f64>
    %57 = fhe.add(%49, %50, %51, %33, %56) : (!fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>, !fhe.batched_secret<8 x f64>) -> !fhe.batched_secret<8 x f64>
    %58 = fhe.extract %57[0] : <8 x f64>
    return %58 : !fhe.secret<f64>
  }
}

