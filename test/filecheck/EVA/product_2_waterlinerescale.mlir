//RUN:  heco --evawaterlinerescale --canonicalize < %s | FileCheck %s

module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.multiply(%0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %2 = eva.multiply(%arg1, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %3 = eva.multiply(%1, %2) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %3 : !eva.cipher<4 x fixed_point>
  }
}

// CHECK: module {
// CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
// CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %1 = eva.multiply(%0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %2 = eva.rescale(%1) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %3 = eva.multiply(%arg1, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %4 = eva.multiply(%2, %3) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     %5 = eva.rescale(%4) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
// CHECK:     return %5 : !eva.cipher<4 x fixed_point>
// CHECK:   }
// CHECK: }
