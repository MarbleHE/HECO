module {
  func.func private @encryptedEigenfaces(%arg0: !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, %arg1: !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> {
    %0 = linalg.init_tensor [4] : tensor<4x!fhe.secret<f64>>
    %1 = bgv.multiply(%arg1, %arg1) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %2 = bgv.multiply(%arg0, %arg0) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %3 = bgv.materialize(%2) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %4 = bgv.sub(%1, %3) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %5 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %6 = bgv.rotate(%4, %5) {offset = 2 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %7 = bgv.add(%4, %6) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %8 = bgv.rotate(%7, %5) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %9 = bgv.add(%7, %8) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %10 = bgv.rotate(%9, %5) {offset = 3 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %11 = bgv.extract %10[0] : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %12 = bgv.materialize(%0) : (tensor<4x!fhe.secret<f64>>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %13 = bgv.insert %11 into %12[0] : (!fhe.secret<f64>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %14 = bgv.rotate(%3, %5) {offset = 4 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %15 = bgv.sub(%1, %14) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %16 = bgv.rotate(%15, %5) {offset = 2 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %17 = bgv.add(%15, %16) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %18 = bgv.rotate(%17, %5) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %19 = bgv.add(%17, %18) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %20 = bgv.rotate(%19, %5) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %21 = bgv.extract %20[1] : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %22 = bgv.insert %21 into %13[1] : (!fhe.secret<f64>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %23 = bgv.rotate(%3, %5) {offset = 8 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %24 = bgv.sub(%1, %23) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %25 = bgv.rotate(%24, %5) {offset = 2 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %26 = bgv.add(%24, %25) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %27 = bgv.rotate(%26, %5) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %28 = bgv.add(%26, %27) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %29 = bgv.extract %28[2] : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %30 = bgv.insert %29 into %22[2] : (!fhe.secret<f64>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %31 = bgv.rotate(%3, %5) {offset = 12 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %32 = bgv.sub(%1, %31) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %33 = bgv.rotate(%32, %5) {offset = 2 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %34 = bgv.add(%32, %33) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %35 = bgv.rotate(%34, %5) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %36 = bgv.add(%34, %35) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %37 = bgv.rotate(%36, %5) {offset = 15 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %38 = bgv.extract %37[3] : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> -> !fhe.secret<f64>
    %39 = bgv.insert %38 into %30[3] : (!fhe.secret<f64>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    return %39 : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
  }
}

