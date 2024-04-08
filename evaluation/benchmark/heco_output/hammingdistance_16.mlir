module {
  func.func private @encryptedHammingDistance_16(%arg0: !emitc.opaque<"seal::Ciphertext">, %arg1: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = emitc.call "evaluator_sub"(%arg0, %arg1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %1 = emitc.call "evaluator_multiply"(%0, %0) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %2 = emitc.call "evaluator_rotate"(%1) {args = [0 : index, 8 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %3 = emitc.call "evaluator_add"(%1, %2) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %4 = emitc.call "evaluator_rotate"(%3) {args = [0 : index, 4 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %5 = emitc.call "evaluator_add"(%3, %4) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %6 = emitc.call "evaluator_rotate"(%5) {args = [0 : index, 2 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %7 = emitc.call "evaluator_add"(%5, %6) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %8 = emitc.call "evaluator_rotate"(%7) {args = [0 : index, 1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %9 = emitc.call "evaluator_add"(%7, %8) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %10 = emitc.call "evaluator_rotate"(%9) {args = [0 : index, 14 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    return %10 : !emitc.opaque<"seal::Ciphertext">
  }
}

