module {
  func.func private @encryptedGxKernel_64x64(%arg0: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %c-2_i16 = arith.constant -2 : i16
    %c2_i16 = arith.constant 2 : i16
    %c-1_i16 = arith.constant -1 : i16
    %0 = emitc.call "evaluator_encode"(%c-2_i16) : (i16) -> !emitc.opaque<"seal::Plaintext">
    %1 = emitc.call "evaluator_multiply_plain"(%arg0, %0) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Plaintext">) -> !emitc.opaque<"seal::Ciphertext">
    %2 = emitc.call "evaluator_encode"(%c-1_i16) : (i16) -> !emitc.opaque<"seal::Plaintext">
    %3 = emitc.call "evaluator_multiply_plain"(%arg0, %2) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Plaintext">) -> !emitc.opaque<"seal::Ciphertext">
    %4 = emitc.call "evaluator_encode"(%c2_i16) : (i16) -> !emitc.opaque<"seal::Plaintext">
    %5 = emitc.call "evaluator_multiply_plain"(%arg0, %4) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Plaintext">) -> !emitc.opaque<"seal::Ciphertext">
    %6 = emitc.call "evaluator_rotate"(%3) {args = [0 : index, 4095 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %7 = emitc.call "evaluator_rotate"(%5) {args = [0 : index, 63 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %8 = emitc.call "evaluator_rotate"(%3) {args = [0 : index, 64 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %9 = emitc.call "evaluator_rotate"(%arg0) {args = [0 : index, 65 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %10 = emitc.call "evaluator_rotate"(%1) {args = [0 : index, 1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %11 = emitc.call "std::vector"() {template_args = [#emitc.opaque<"seal::Ciphertext">]} : () -> !emitc.opaque<"std::vector<seal::Ciphertext>">
    emitc.call "insert"(%11, %6) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %7) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %arg0) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %8) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %9) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %10) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    %12 = emitc.call "evaluator_add_many"(%11) : (!emitc.opaque<"std::vector<seal::Ciphertext>">) -> !emitc.opaque<"seal::Ciphertext">
    return %12 : !emitc.opaque<"seal::Ciphertext">
  }
}

