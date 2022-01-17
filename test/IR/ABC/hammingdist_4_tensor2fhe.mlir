//RUN: abc-opt -batching --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
    %0 = fhe.extract %arg0[0] : <f64>
    %1 = fhe.extract %arg1[0] : <f64>
    %2 = fhe.sub(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %3 = fhe.multiply(%2, %2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %4 = fhe.extract %arg0[1] : <f64>
    %5 = fhe.extract %arg1[1] : <f64>
    %6 = fhe.sub(%4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %7 = fhe.multiply(%6, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %8 = fhe.extract %arg0[2] : <f64>
    %9 = fhe.extract %arg1[2] : <f64>
    %10 = fhe.sub(%8, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %11 = fhe.multiply(%10, %10) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %12 = fhe.extract %arg0[3] : <f64>
    %13 = fhe.extract %arg1[3] : <f64>
    %14 = fhe.sub(%12, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %15 = fhe.multiply(%14, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %16 = fhe.add(%15, %11, %3, %7) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    return %16 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
// CHECK:     %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %2 = fhe.rotate(%1) {i = -3 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %3 = fhe.rotate(%1) {i = -2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %4 = fhe.rotate(%1) {i = -1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %5 = fhe.add(%2, %3, %1, %4) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %6 = fhe.extract %5[0] : <f64>
// CHECK:     return %6 : !fhe.secret<f64>
// CHECK:   }
// CHECK: }