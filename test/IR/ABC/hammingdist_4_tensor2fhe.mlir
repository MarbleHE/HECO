//RUN: abc-opt -batching --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
    %c0_sf64 = fhe.constant 0.000000e+00 : f64
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
    %16 = fhe.add(%15, %11, %7, %3, %c0_sf64) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    return %16 : !fhe.secret<f64>
  }
}