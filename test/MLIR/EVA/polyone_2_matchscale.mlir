//RUN:  fhe-tool -matchscale --canonicalize < %s | FileCheck %s
module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
    %1 = eva.add(%0, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
    return %1 : !eva.cipher<4 x fixed_point : 60 : -1>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     %1 = eva.constant <1, 1, 1, 1> : !eva.vector<4 x fixed_point : 30>
//CHECK:     %2 = eva.multiply(%arg0, %1): (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.vector<4 x fixed_point : 30 : -1>) -> eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     %3 = eva.add(%2, %3) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     return %3 : !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:   }
//CHECK: }
