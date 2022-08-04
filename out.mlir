module {
  func private @encryptedBoxBlur(%arg0: !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> {
    %0 = bgv.materialize(%arg0) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %1 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %2 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %3 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %4 = bgv.rotate(%2, %3) {offset = 55 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %5 = bgv.materialize(%4) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %6 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %7 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %8 = bgv.rotate(%6, %7) {offset = 63 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %9 = bgv.materialize(%8) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %10 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %11 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %12 = bgv.rotate(%10, %11) {offset = 7 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %13 = bgv.materialize(%12) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %14 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %15 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %16 = bgv.rotate(%14, %15) {offset = 56 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %17 = bgv.materialize(%16) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %18 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %19 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %20 = bgv.rotate(%18, %19) {offset = 8 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %21 = bgv.materialize(%20) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %22 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %23 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %24 = bgv.rotate(%22, %23) {offset = 57 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %25 = bgv.materialize(%24) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %26 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %27 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %28 = bgv.rotate(%26, %27) {offset = 9 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %29 = bgv.materialize(%28) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %30 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %31 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %32 = bgv.rotate(%30, %31) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %33 = bgv.materialize(%32) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %34 = bgv.materialize(%5) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %35 = bgv.materialize(%9) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %36 = bgv.materialize(%13) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %37 = bgv.materialize(%17) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %38 = bgv.materialize(%0) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %39 = bgv.materialize(%21) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %40 = bgv.materialize(%25) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %41 = bgv.materialize(%29) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %42 = bgv.materialize(%33) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %43 = bgv.add_many(%34, %35, %36, %37, %38, %39, %40, %41, %42) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %44 = bgv.materialize(%43) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !fhe.batched_secret<64 x f64>
    %45 = bgv.materialize(%44) : (!fhe.batched_secret<64 x f64>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    return %45 : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
  }
}

