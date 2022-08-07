//RUN:  fhe-tool -fhe2bgv --canonicalize --cse < %s | FileCheck %s
module {
  func.func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64> {
    %0 = fhe.rotate(%arg0) by 55 : <64 x f64>
    %1 = fhe.rotate(%arg0) by 63 : <64 x f64>
    %2 = fhe.rotate(%arg0) by 7 : <64 x f64>
    %3 = fhe.rotate(%arg0) by 56 : <64 x f64>
    %4 = fhe.rotate(%arg0) by 8 : <64 x f64>
    %5 = fhe.rotate(%arg0) by 57 : <64 x f64>
    %6 = fhe.rotate(%arg0) by 9 : <64 x f64>
    %7 = fhe.rotate(%arg0) by 1 : <64 x f64>
    %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
    return %8 : !fhe.batched_secret<64 x f64>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedBoxBlur(%arg0: !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">> {
//CHECK:     %0 = bgv.load_galois_keys {file = "foo.glk", parms = "glk.parms"} : !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %1 = bgv.rotate(%arg0, %0) {offset = 55 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %2 = bgv.rotate(%arg0, %0) {offset = 63 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %3 = bgv.rotate(%arg0, %0) {offset = 7 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %4 = bgv.rotate(%arg0, %0) {offset = 56 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %5 = bgv.rotate(%arg0, %0) {offset = 8 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %6 = bgv.rotate(%arg0, %0) {offset = 57 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %7 = bgv.rotate(%arg0, %0) {offset = 9 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %8 = bgv.rotate(%arg0, %0) {offset = 1 : i64} : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.glk<0 x 0 x 0 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     %9 = bgv.add_many(%1, %2, %3, %4, %arg0, %5, %6, %7, %8) : (!bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:     return %9 : !bgv.ctxt<1 x !poly.poly<2, true, 17, "parms.txt">>
//CHECK:   }
//CHECK: }