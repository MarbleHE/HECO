module {
  func.func private @encryptedPSU(%arg0: !fhe.batched_secret<8 x i16>, %arg1: !fhe.batched_secret<4 x i16>, %arg2: !fhe.batched_secret<8 x i16>, %arg3: !fhe.batched_secret<4 x i16>) -> !fhe.secret<i16> {
    %cst = fhe.constant dense<1> : tensor<8xi16>
    %0 = fhe.materialize(%arg1) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>
    %1 = fhe.sub(%arg0, %arg2) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %2 = fhe.multiply(%1, %1) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %3 = fhe.sub(%cst, %2) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %4 = fhe.rotate(%3) by 1 : <8 x i16>
    %5 = fhe.multiply(%3, %4) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %6 = fhe.sub(%cst, %5) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %7 = fhe.rotate(%arg2) by 6 : <8 x i16>
    %8 = fhe.sub(%arg0, %7) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %9 = fhe.multiply(%8, %8) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %10 = fhe.sub(%cst, %9) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %11 = fhe.rotate(%10) by 1 : <8 x i16>
    %12 = fhe.multiply(%10, %11) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %13 = fhe.sub(%cst, %12) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %14 = fhe.rotate(%arg2) by 4 : <8 x i16>
    %15 = fhe.sub(%arg0, %14) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %16 = fhe.multiply(%15, %15) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %17 = fhe.sub(%cst, %16) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %18 = fhe.rotate(%17) by 1 : <8 x i16>
    %19 = fhe.multiply(%17, %18) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %20 = fhe.sub(%cst, %19) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %21 = fhe.rotate(%arg2) by 2 : <8 x i16>
    %22 = fhe.sub(%arg0, %21) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %23 = fhe.multiply(%22, %22) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %24 = fhe.sub(%cst, %23) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %25 = fhe.rotate(%24) by 1 : <8 x i16>
    %26 = fhe.multiply(%24, %25) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %27 = fhe.sub(%cst, %26) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %28 = fhe.materialize(%arg3) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>
    %29 = fhe.rotate(%27) by 7 : <8 x i16>
    %30 = fhe.rotate(%20) by 7 : <8 x i16>
    %31 = fhe.rotate(%6) by 7 : <8 x i16>
    %32 = fhe.rotate(%13) by 7 : <8 x i16>
    %33 = fhe.multiply(%28, %29, %30, %31, %32) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %34 = fhe.rotate(%20) by 6 : <8 x i16>
    %35 = fhe.rotate(%13) by 6 : <8 x i16>
    %36 = fhe.rotate(%27) by 6 : <8 x i16>
    %37 = fhe.rotate(%6) by 6 : <8 x i16>
    %38 = fhe.multiply(%28, %34, %35, %36, %37) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %39 = fhe.rotate(%13) by 5 : <8 x i16>
    %40 = fhe.rotate(%6) by 5 : <8 x i16>
    %41 = fhe.rotate(%20) by 5 : <8 x i16>
    %42 = fhe.rotate(%27) by 5 : <8 x i16>
    %43 = fhe.multiply(%28, %39, %40, %41, %42) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %44 = fhe.rotate(%6) by 4 : <8 x i16>
    %45 = fhe.rotate(%27) by 4 : <8 x i16>
    %46 = fhe.rotate(%13) by 4 : <8 x i16>
    %47 = fhe.rotate(%20) by 4 : <8 x i16>
    %48 = fhe.multiply(%28, %44, %45, %46, %47) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %49 = fhe.rotate(%48) by 5 : <8 x i16>
    %50 = fhe.rotate(%43) by 6 : <8 x i16>
    %51 = fhe.rotate(%38) by 7 : <8 x i16>
    %52 = fhe.rotate(%0) by 2 : <8 x i16>
    %53 = fhe.add(%0, %52) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %54 = fhe.rotate(%53) by 1 : <8 x i16>
    %55 = fhe.add(%53, %54) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %56 = fhe.rotate(%55) by 3 : <8 x i16>
    %57 = fhe.add(%49, %50, %51, %33, %56) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %58 = fhe.extract %57[0] : <8 x i16>
    return %58 : !fhe.secret<i16>
  }
}

