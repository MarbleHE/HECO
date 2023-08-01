module {
  func.func private @encryptedBoxBlur(%arg0: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = "emitc.constant"() {value = #emitc.opaque<"\22glk.parms\22">} : () -> !emitc.opaque<"std::string">
    %1 = "emitc.constant"() {value = #emitc.opaque<"\22foo.glk\22">} : () -> !emitc.opaque<"std::string">
    %2 = emitc.call "evaluator_load_galois_keys"(%1, %0) : (!emitc.opaque<"std::string">, !emitc.opaque<"std::string">) -> !emitc.opaque<"seal::GaloisKeys">
    %3 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 55 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %4 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 63 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %5 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 7 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %6 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 56 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %7 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 8 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %8 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 57 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %9 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 9 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %10 = emitc.call "evaluator_rotate"(%arg0, %2) {args = [0 : index, 1 : si32, 1 : index]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::GaloisKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %11 = emitc.call "std::vector"() {template_args = [#emitc.opaque<"seal::Ciphertext">]} : () -> !emitc.opaque<"std::vector<seal::Ciphertext>">
    emitc.call "insert"(%11, %3) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %4) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %5) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %6) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %arg0) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %7) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %8) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %9) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    emitc.call "insert"(%11, %10) : (!emitc.opaque<"std::vector<seal::Ciphertext>">, !emitc.opaque<"seal::Ciphertext">) -> ()
    %12 = emitc.call "evaluator_add_many"(%11) : (!emitc.opaque<"std::vector<seal::Ciphertext>">) -> !emitc.opaque<"seal::Ciphertext">
    return %12 : !emitc.opaque<"seal::Ciphertext">
  }
}

