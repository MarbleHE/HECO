module {
  func.func private @encryptedDotProduct_8(%arg0: !emitc.opaque<"seal::Ciphertext">, %arg1: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = emitc.call "evaluator_multiply"(%arg0, %arg1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %1 = emitc.call "evaluator_rotate"(%0) {args = [0 : index, 2 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %2 = emitc.call "evaluator_add"(%0, %1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %3 = emitc.call "evaluator_rotate"(%2) {args = [0 : index, 1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %4 = emitc.call "evaluator_add"(%2, %3) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %5 = emitc.call "evaluator_rotate"(%4) {args = [0 : index, 3 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    return %5 : !emitc.opaque<"seal::Ciphertext">
  }
}

