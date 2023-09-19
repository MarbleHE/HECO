//RUN:  fhe-tool -eagermodswitch --canonicalize < %s | FileCheck %s
module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 120 : -1>
    %1 = eva.rescale(%0) by 60 : !eva.cipher<4 x fixed_point : 120 : -1> -> !eva.cipher<4 x fixed_point : 60 : -1>
    %2 = eva.add(%1, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
    %3 = eva.add(%2, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
    return %3 : !eva.cipher<4 x fixed_point : 60 : -1>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 120 : -1>
//CHECK:     %1 = eva.rescale(%0) by 60 : !eva.cipher<4 x fixed_point : 120 : -1> -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     %2 = eva.modswitch(%arg0) : !eva.cipher<4 x fixed_point : 60 : -1> -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     %4 = eva.add(%3, %2) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:     return %4 : !eva.cipher<4 x fixed_point : 60 : -1>
//CHECK:   }
//CHECK: }
