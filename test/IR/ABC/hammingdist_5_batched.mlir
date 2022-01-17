//RUN: abc-opt -internal-batching --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
    %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %2 = fhe.rotate(%1) {i = -1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %3 = fhe.rotate(%1) {i = -2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %4 = fhe.rotate(%1) {i = -3 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %5 = fhe.add(%1, %2, %3, %4) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %6 = fhe.extract %5[0] : <f64>
    return %6 : !fhe.secret<f64>
  }
}