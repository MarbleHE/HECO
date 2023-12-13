//RUN:  heco --evawaterline="source_scale=60 scale_drop=60 waterline=60" --canonicalize < %s | FileCheck %s
module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.add(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %2 = eva.add(%0, %1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %2 : !eva.cipher<4 x fixed_point>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.rescale(%0) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.add(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %3 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
