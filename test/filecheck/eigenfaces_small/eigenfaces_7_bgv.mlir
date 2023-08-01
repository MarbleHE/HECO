module {
  func.func private @encryptedEigenfaces(%arg0: !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, %arg1: !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> {
    %0 = linalg.init_tensor [4] : tensor<4x!fhe.secret<f64>>
    %1 = bfv.multiply(%arg1, %arg1) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %2 = bfv.multiply(%arg0, %arg0) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %3 = bfv.materialize(%2) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %4 = bfv.sub(%1, %3) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %5 = bfv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %6 = bfv.rotate(%4, %5) {offset = 2 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %7 = bfv.add(%4, %6) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %8 = bfv.rotate(%7, %5) {offset = 1 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %9 = bfv.add(%7, %8) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %10 = bfv.rotate(%9, %5) {offset = 3 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %11 = bfv.extract %10[0] : !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %12 = bfv.materialize(%0) : (tensor<4x!fhe.secret<f64>>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %13 = bfv.insert %11 into %12[0] : (!fhe.secret<f64>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %14 = bfv.rotate(%3, %5) {offset = 4 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %15 = bfv.sub(%1, %14) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %16 = bfv.rotate(%15, %5) {offset = 2 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %17 = bfv.add(%15, %16) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %18 = bfv.rotate(%17, %5) {offset = 1 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %19 = bfv.add(%17, %18) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %20 = bfv.rotate(%19, %5) {offset = 1 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %21 = bfv.extract %20[1] : !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %22 = bfv.insert %21 into %13[1] : (!fhe.secret<f64>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %23 = bfv.rotate(%3, %5) {offset = 8 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %24 = bfv.sub(%1, %23) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %25 = bfv.rotate(%24, %5) {offset = 2 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %26 = bfv.add(%24, %25) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %27 = bfv.rotate(%26, %5) {offset = 1 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %28 = bfv.add(%26, %27) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %29 = bfv.extract %28[2] : !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %30 = bfv.insert %29 into %22[2] : (!fhe.secret<f64>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %31 = bfv.rotate(%3, %5) {offset = 12 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %32 = bfv.sub(%1, %31) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %33 = bfv.rotate(%32, %5) {offset = 2 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %34 = bfv.add(%32, %33) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %35 = bfv.rotate(%34, %5) {offset = 1 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %36 = bfv.add(%34, %35) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %37 = bfv.rotate(%36, %5) {offset = 15 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %38 = bfv.extract %37[3] : !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %39 = bfv.insert %38 into %30[3] : (!fhe.secret<f64>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    return %39 : !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
  }
}

