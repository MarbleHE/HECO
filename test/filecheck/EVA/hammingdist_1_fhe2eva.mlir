//RUN:  heco --fhe2eva --canonicalize < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<4 x f64>, %arg1: !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %2 = fhe.rotate(%1) by -2 : <4 x f64>
    %3 = fhe.add(%1, %2) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %4 = fhe.rotate(%3) by -1 : <4 x f64>
    %5 = fhe.add(%3, %4) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    return %5 : !fhe.batched_secret<4 x f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
// CHECK:     %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %2 = eva.rotate(%1) by 2 : <4 x fixed_point>
// CHECK:     %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %4 = eva.rotate(%3) by 3 : <4 x fixed_point>
// CHECK:     %5 = eva.add(%3, %4) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     return %5 : !eva.cipher<4 x fixed_point>
// CHECK:   }
// CHECK: }