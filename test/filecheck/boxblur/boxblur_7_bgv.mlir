module {
  func.func private @encryptedBoxBlur(%arg0: !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> {
    %0 = bfv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %1 = bfv.rotate(%arg0, %0) {offset = 55 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %2 = bfv.rotate(%arg0, %0) {offset = 63 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %3 = bfv.rotate(%arg0, %0) {offset = 7 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %4 = bfv.rotate(%arg0, %0) {offset = 56 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %5 = bfv.rotate(%arg0, %0) {offset = 8 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %6 = bfv.rotate(%arg0, %0) {offset = 57 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %7 = bfv.rotate(%arg0, %0) {offset = 9 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %8 = bfv.rotate(%arg0, %0) {offset = 1 : i64} : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %9 = bfv.add_many(%1, %2, %3, %4, %arg0, %5, %6, %7, %8) : (!bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    return %9 : !bfv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
  }
}

