//RUN:  fhe-tool -fhe2eva --canonicalize < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<4 x f64>, %arg1: !fhe.batched_secret<4 x f64>) -> !fhe.secret<f64> {
    %0 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %1 = fhe.multiply(%0, %0) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %2 = fhe.rotate(%1) by -2 : <4 x f64>
    %3 = fhe.add(%1, %2) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %4 = fhe.rotate(%3) by -1 : <4 x f64>
    %5 = fhe.add(%3, %4) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %6 = fhe.extract %5[0] : <4 x f64>
    return %6 : !fhe.secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point : 30 : -1>, %arg1: !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1> {
// CHECK:     %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 30 : -1>
// CHECK:     %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
// CHECK:     %2 = eva.rotate(%1) by -2 : <4 x fixed_point : 60 : -1>
// CHECK:     %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
// CHECK:     %4 = eva.rotate(%3) by -1 : <4 x fixed_point : 60 : -1>
// CHECK:     %5 = eva.add(%3, %4) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
// CHECK:     return %5 : !eva.cipher<4 x fixed_point : 60 : -1>
// CHECK:   }
// CHECK: }