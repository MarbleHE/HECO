//RUN:  fhe-tool -fhe2eva --canonicalize < %s | FileCheck %s
module  {
  func.func private @encryptedPoly(%arg0: !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64> {
    %0 = fhe.multiply(%arg0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %1 = fhe.add(%0, %arg0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    return %1 : !fhe.batched_secret<4 x f64>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point :30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     %1 = eva.add(%0, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     return %1 : !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:   }
//CHECK: }
