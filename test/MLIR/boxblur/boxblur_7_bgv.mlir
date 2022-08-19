//RUN:  fhe-tool -bgv2emitc --canonicalize --cse < %s | FileCheck %s
module {
  func.func private @encryptedBoxBlur(%arg0: !bgv.ctxt<1 x !poly.poly<2, 1, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, 1, 17, "parms.txt">> {
     %9 = bgv.add_many(%arg0, %arg0) : (!bgv.ctxt<1 x !poly.poly<2, 1, 17, "parms.txt">>, !bgv.ctxt<1 x !poly.poly<2, 1, 17, "parms.txt">>) -> !bgv.ctxt<1 x !poly.poly<2, 1, 17, "parms.txt">>
    return %9 : !bgv.ctxt<1 x !poly.poly<2, 1, 17, "parms.txt">>
  }
}

// //TODO: Figure out why the bgv2emitc pass has started crashing!
// func.func private @f(%x: !emitc.opaque<"seal::ciphertext">, %y:!emitc.opaque<"seal::ciphertext"> ) -> !emitc.opaque<"seal::ciphertext"> {
//   %0 = "emitc.constant"() {value = "foo.glk"} :() -> (!emitc.opaque<"std::string">)
//   %1 = emitc.call "evaluator.load_glk" (%0) {} : (!emitc.opaque<"std::string">) -> (!emitc.opaque<"seal::ciphertext">)
//   return %1 : !emitc.opaque<"seal::ciphertext">
// }