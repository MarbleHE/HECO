//RUN: fhe-tool -internal-batching --canonicalize --cse < %s | FileCheck %s
module {
  func.func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64> {
    %0 = fhe.rotate(%arg0) by 55 : <64 x f64>
    %1 = fhe.rotate(%arg0) by 63 : <64 x f64>
    %2 = fhe.rotate(%arg0) by 7 : <64 x f64>
    %3 = fhe.rotate(%arg0) by 56 : <64 x f64>
    %4 = fhe.rotate(%arg0) by 8 : <64 x f64>
    %5 = fhe.rotate(%arg0) by 57 : <64 x f64>
    %6 = fhe.rotate(%arg0) by 9 : <64 x f64>
    %7 = fhe.rotate(%arg0) by 1 : <64 x f64>
    %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
    return %8 : !fhe.batched_secret<64 x f64>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64> {
//CHECK:     %0 = fhe.rotate(%arg0) by 55 : <64 x f64>
//CHECK:     %1 = fhe.rotate(%arg0) by 63 : <64 x f64>
//CHECK:     %2 = fhe.rotate(%arg0) by 7 : <64 x f64>
//CHECK:     %3 = fhe.rotate(%arg0) by 56 : <64 x f64>
//CHECK:     %4 = fhe.rotate(%arg0) by 8 : <64 x f64>
//CHECK:     %5 = fhe.rotate(%arg0) by 57 : <64 x f64>
//CHECK:     %6 = fhe.rotate(%arg0) by 9 : <64 x f64>
//CHECK:     %7 = fhe.rotate(%arg0) by 1 : <64 x f64>
//CHECK:     %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
//CHECK:     return %8 : !fhe.batched_secret<64 x f64>
//CHECK:   }
//CHECK: }