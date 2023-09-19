//RUN:  fhe-tool -waterlinerescale --canonicalize < %s | FileCheck %s
module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 30 : -1>, %arg1: !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 210 : -1> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
    %1 = eva.multiply(%0, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 90 : -1>
    %2 = eva.multiply(%arg1, %arg1) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 120 : -1>
    %3 = eva.multiply(%1, %2) : (!eva.cipher<4 x fixed_point : 90 : -1>, !eva.cipher<4 x fixed_point : 120 : -1>) -> !eva.cipher<4 x fixed_point : 210 : -1>
    return %3 : !eva.cipher<4 x fixed_point : 210 : -1>
  }
}

// CHECK: module {
// CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point : 30 : -1>, %arg1: !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 90 : -1> {
// CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 60 : -1>
// CHECK:     %1 = eva.multiply(%0, %arg0) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 30 : -1>) -> !eva.cipher<4 x fixed_point : 90 : -1>
// CHECK:     %2 = eva.rescale(%1) by 60 : !eva.cipher<4 x fixed_point : 90 : -1> -> !eva.cipher<4 x fixed_point : 30 : -1>
// CHECK:     %3 = eva.multiply(%arg1, %arg1) : (!eva.cipher<4 x fixed_point : 60 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 120 : -1>
// CHECK:     %4 = eva.rescale(%3) by 60 : !eva.cipher<4 x fixed_point : 120 : -1> -> !eva.cipher<4 x fixed_point : 60 : -1>
// CHECK:     %5 = eva.multiply(%2, %4) : (!eva.cipher<4 x fixed_point : 30 : -1>, !eva.cipher<4 x fixed_point : 60 : -1>) -> !eva.cipher<4 x fixed_point : 90 : -1>
// CHECK:     return %5 : !eva.cipher<4 x fixed_point : 90 : -1>
// CHECK:   }
// CHECK: }
