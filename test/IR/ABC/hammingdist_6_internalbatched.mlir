//RUN:  abc-opt -fhe2emitc --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
    %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %2 = fhe.rotate(%1) {i = -2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %3 = fhe.add(%1, %2) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %4 = fhe.rotate(%3) {i = -1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %5 = fhe.add(%3, %4) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %6 = fhe.extract %5[0] : <f64>
    return %6 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: !emitc.opaque<"seal::Ciphertext">, %arg1: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
// CHECK:     %0 = emitc.call "evaluator.sub"(%arg0, %arg1) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %1 = emitc.call "evaluator.multiply"(%0, %0) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %2 = emitc.call "evaluator.rotate"(%1) {args = [0 : index, -2 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %3 = emitc.call "evaluator.add"(%1, %2) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %4 = emitc.call "evaluator.rotate"(%3) {args = [0 : index, -1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %5 = emitc.call "evaluator.add"(%3, %4) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %6 = emitc.call "evaluator.rotate"(%5) {args = [0 : index, 0 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     return %6 : !emitc.opaque<"seal::Ciphertext">
// CHECK:   }
// CHECK: }
