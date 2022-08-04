//RUN:  abc-opt -fhe2emitc --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64> {
    %0 = fhe.rotate(%arg0) by -9 : <f64>
    %1 = fhe.rotate(%arg0) by -1 : <f64>
    %2 = fhe.rotate(%arg0) by -57 : <f64>
    %3 = fhe.rotate(%arg0) by -8 : <f64>
    %4 = fhe.rotate(%arg0) by -56 : <f64>
    %5 = fhe.rotate(%arg0) by -7 : <f64>
    %6 = fhe.rotate(%arg0) by -55 : <f64>
    %7 = fhe.rotate(%arg0) by -63 : <f64>
    %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %9 = fhe.rotate(%arg0) by 1 : <f64>
    %10 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %11 = fhe.rotate(%arg0) by 7 : <f64>
    %12 = fhe.add(%0, %1, %11, %3, %arg0, %4, %5, %6, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %13 = fhe.rotate(%arg0) by 8 : <f64>
    %14 = fhe.add(%0, %1, %11, %3, %arg0, %13, %5, %6, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %15 = fhe.rotate(%arg0) by 9 : <f64>
    %16 = fhe.add(%0, %1, %11, %3, %arg0, %13, %5, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %17 = fhe.rotate(%arg0) by 55 : <f64>
    %18 = fhe.add(%17, %1, %11, %3, %arg0, %13, %5, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %19 = fhe.rotate(%arg0) by 56 : <f64>
    %20 = fhe.add(%17, %1, %11, %19, %arg0, %13, %5, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %21 = fhe.rotate(%arg0) by 57 : <f64>
    %22 = fhe.add(%17, %1, %11, %19, %arg0, %13, %21, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %23 = fhe.rotate(%arg0) by 63 : <f64>
    %24 = fhe.add(%17, %23, %11, %19, %arg0, %13, %21, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %25 = fhe.combine(%8[0], %10[1:6], %12[7], %14[8], %16[9:54], %18[55], %20[56], %22[57:62], %24[63], %arg0) : !fhe.batched_secret<f64>
    return %25 : !fhe.batched_secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedBoxBlur(%arg0: !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext"> {
// CHECK:     %0 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -9 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %1 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %2 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -57 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %3 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -8 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %4 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -56 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %5 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -7 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %6 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -55 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %7 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, -63 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %8 = emitc.call "evaluator.add_many"(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %9 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 1 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %10 = emitc.call "evaluator.add_many"(%0, %1, %2, %3, %arg0, %4, %5, %6, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %11 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 7 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %12 = emitc.call "evaluator.add_many"(%0, %1, %11, %3, %arg0, %4, %5, %6, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %13 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 8 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %14 = emitc.call "evaluator.add_many"(%0, %1, %11, %3, %arg0, %13, %5, %6, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %15 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 9 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %16 = emitc.call "evaluator.add_many"(%0, %1, %11, %3, %arg0, %13, %5, %15, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %17 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 55 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %18 = emitc.call "evaluator.add_many"(%17, %1, %11, %3, %arg0, %13, %5, %15, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %19 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 56 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %20 = emitc.call "evaluator.add_many"(%17, %1, %11, %19, %arg0, %13, %5, %15, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %21 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 57 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %22 = emitc.call "evaluator.add_many"(%17, %1, %11, %19, %arg0, %13, %21, %15, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %23 = emitc.call "evaluator.rotate"(%arg0) {args = [0 : index, 63 : si32]} : (!emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %24 = emitc.call "evaluator.add_many"(%17, %23, %11, %19, %arg0, %13, %21, %15, %9) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     %25 = emitc.call "evaluator.combine"(%8, %10, %12, %14, %16, %18, %20, %22, %24, %arg0) {args = [0 : index, #emitc.opaque<"{0}">, 1 : index, #emitc.opaque<"{1, 2, 3, 4, 5, 6}">, 2 : index, #emitc.opaque<"{7}">, 3 : index, #emitc.opaque<"{8}">, 4 : index, #emitc.opaque<"{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54}">, 5 : index, #emitc.opaque<"{55}">, 6 : index, #emitc.opaque<"{56}">, 7 : index, #emitc.opaque<"{57, 58, 59, 60, 61, 62}">, 8 : index, #emitc.opaque<"{63}">, 9 : index, #emitc.opaque<"{}">]} : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
// CHECK:     return %25 : !emitc.opaque<"seal::Ciphertext">
// CHECK:   }
// CHECK: }