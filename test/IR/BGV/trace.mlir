func private @trace() -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">> {
  %0 = bgv.load_public_key {file = "public_key_0.pk", parms = "p_0.parms"} : !bgv.pk<2 x !poly.poly<8192, true, 5, "p_0.parms">>
  %1 = bgv.load_relin_keys {file = "relin_keys_1.rk", parms = "p_0.parms"} : !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>
  %2 = bgv.load_ctxt {file = "encrpytion_2.ctxt", parms = "p_1.parms"} : !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %3 = bgv.multiply(%2, %2) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>
  %4 = bgv.relinearize(%3, %1) : (!bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %5 = bgv.multiply(%4, %4) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>
  %6 = bgv.relinearize(%5, %1) : (!bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %7 = bgv.multiply(%6, %6) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>
  %8 = bgv.relinearize(%7, %1) : (!bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %9 = bgv.load_ctxt {file = "encrpytion_3.ctxt", parms = "p_1.parms"} : !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %10 = bgv.multiply(%9, %9) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>
  %11 = bgv.relinearize(%10, %1) : (!bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %12 = bgv.modswitch_to(%11, %11) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %13 = bgv.multiply(%12, %12) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>
  %14 = bgv.relinearize(%13, %1) : (!bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %15 = bgv.modswitch_to(%14, %14) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %16 = bgv.multiply(%15, %15) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>
  %17 = bgv.relinearize(%16, %1) : (!bgv.ctxt<3 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.rlk<1 x 4 x 2 x !poly.poly<8192, true, 5, "p_0.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  %18 = bgv.modswitch_to(%17, %17) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>, !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>) -> !bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>
  bgv.sink(%15) : (!bgv.ctxt<2 x !poly.poly<8192, false, 4, "p_1.parms">>)
}