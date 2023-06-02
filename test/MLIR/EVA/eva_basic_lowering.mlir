//RUN:  fhe-tool -fhe2eva < %s | FileCheck %s
module {
  func.func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64> {
    %0 = fhe.rotate(%arg0) by 55 : <64 x f64>
    %1 = fhe.rotate(%arg0) by 63 : <64 x f64>
    %2 = fhe.rotate(%arg0) by 7 : <64 x f64>
    %3 = fhe.rotate(%arg0) by 56 : <64 x f64>
    %8 = fhe.add(%0, %1) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
    %9 = fhe.multiply(%2, %3) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
    %10 = fhe.add(%arg0, %8) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
    %11 = fhe.sub(%10, %9) : (!fhe.batched_secret<64 x f64>, !fhe.batched_secret<64 x f64>) -> !fhe.batched_secret<64 x f64>
    return %3 : !fhe.batched_secret<64 x f64>
  }
}