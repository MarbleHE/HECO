//RUN: abc-opt -internal-batching --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
    %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %2 = fhe.rotate(%1) {i = -3 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %3 = fhe.rotate(%1) {i = -2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %4 = fhe.rotate(%1) {i = -1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %5 = fhe.add(%2, %3, %1, %4) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %6 = fhe.extract %5[0] : <f64>
    return %6 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
// CHECK:     %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %2 = fhe.rotate(%1) {i = -2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %3 = fhe.add(%1, %2) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %4 = fhe.rotate(%3) {i = -1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %5 = fhe.add(%3, %4) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %6 = fhe.extract %5[0] : <f64>
// CHECK:     return %6 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }