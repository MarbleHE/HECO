module {
  func.func private @encryptedBoxBlur(%arg0: !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> {
    %0 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
    %1 = bgv.rotate(%arg0, %0) {offset = 55 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %2 = bgv.rotate(%arg0, %0) {offset = 63 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %3 = bgv.rotate(%arg0, %0) {offset = 7 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %4 = bgv.rotate(%arg0, %0) {offset = 56 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %5 = bgv.rotate(%arg0, %0) {offset = 8 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %6 = bgv.rotate(%arg0, %0) {offset = 57 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %7 = bgv.rotate(%arg0, %0) {offset = 9 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %8 = bgv.rotate(%arg0, %0) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    %9 = bgv.add_many(%1, %2, %3, %4, %arg0, %5, %6, %7, %8) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
    return %9 : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
  }
}

