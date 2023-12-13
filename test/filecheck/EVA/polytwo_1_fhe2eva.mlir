//RUN:  heco --fhe2eva --canonicalize < %s | FileCheck %s
module  {
  func.func private @encryptedPoly(%arg0: !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = fhe.multiply(%arg0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %1 = fhe.add(%arg0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %2 = fhe.add(%0, %1) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    return %2 : !fhe.batched_secret<4 x f64>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.add(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.add(%0, %1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %2 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
