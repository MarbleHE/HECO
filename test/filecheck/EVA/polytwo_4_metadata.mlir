//RUN:  heco --evametadata="source_scale=60 scale_drop=60 waterline=60" --canonicalize < %s | FileCheck %s

module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.rescale(%0) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %2 = eva.add(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %3 = eva.modswitch(%2) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %4 = eva.add(%1, %3) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %4 : !eva.cipher<4 x fixed_point>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) {result_mod = 7 : si32, result_scale = 120 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.rescale(%0) {result_mod = 6 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.add(%arg0, %arg0) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %3 = eva.modswitch(%2) {result_mod = 6 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %4 = eva.add(%1, %3) {result_mod = 6 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %4 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
