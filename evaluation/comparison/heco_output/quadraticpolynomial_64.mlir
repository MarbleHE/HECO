module {
  func.func private @encryptedQuadraticPolynomial_64(%arg0: !emitc.opaque<"seal::Ciphertext">, %arg1: !emitc.opaque<"seal::Ciphertext">, %arg2: !emitc.opaque<"seal::Ciphertext">, %arg3: !emitc.opaque<"seal::Ciphertext">, %arg4: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = emitc.call "evaluator_multiply"(%arg0, %arg3) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %1 = emitc.call "evaluator_add"(%0, %arg1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %2 = emitc.call "evaluator_multiply"(%arg3, %1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %3 = emitc.call "evaluator_add"(%2, %arg2) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %4 = emitc.call "evaluator_sub"(%arg4, %3) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    return %4 : !emitc.opaque<"seal::Ciphertext">
  }
}

