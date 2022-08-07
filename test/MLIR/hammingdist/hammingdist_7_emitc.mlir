//RUN:  emitc-translate -mlir-to-cpp < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: !emitc.opaque<"seal::Ciphertext">, %arg1: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = emitc.call "evaluator.sub"(%arg0, %arg1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %1 = emitc.call "evaluator.multiply"(%0, %0) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %2 = emitc.call "evaluator.rotate"(%1) {args = [0 : index, -2 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %3 = emitc.call "evaluator.add"(%1, %2) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %4 = emitc.call "evaluator.rotate"(%3) {args = [0 : index, -1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %5 = emitc.call "evaluator.add"(%3, %4) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %6 = emitc.call "evaluator.rotate"(%5) {args = [0 : index, 0 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    return %6 : !emitc.opaque<"seal::Ciphertext">
  }
}

// see hammingdist_8_cpp.cpp for expected output