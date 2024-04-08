module {
  func.func private @encryptedBoxBlur_64x64(%arg0: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = emitc.call "evaluator_rotate"(%arg0) {args = [0 : index, 64 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %1 = emitc.call "evaluator_rotate"(%arg0) {args = [0 : index, 65 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %2 = emitc.call "evaluator_rotate"(%arg0) {args = [0 : index, 1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %3 = emitc.call "std::vector"() {template_args = [#emitc.opaque<"seal::Ciphertext">]} : () -> !emitc.opaque<"std::vector<seal::Ciphertext>">
    emitc.call "insert"(%3, %arg0) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%3, %0) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%3, %1) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%3, %2) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    %4 = emitc.call "evaluator_add_many"(%3) : (!emitc.opaque<"std::vector<seal::Ciphertext>">) -> !emitc.opaque<"seal::Ciphertext">
    return %4 : !emitc.opaque<"seal::Ciphertext">
  }
}

