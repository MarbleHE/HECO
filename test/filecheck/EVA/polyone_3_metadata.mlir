//RUN:  heco --evametadata --canonicalize < %s | FileCheck %s

module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.constant() {result_mod = -1 : si32, result_scale = 30 : si32, value = [1 : i32, 1 : i32, 1 : i32, 1 : i32]} : (none) -> !eva.cipher<4 x fixed_point>
    %2 = eva.multiply(%arg0, %1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %3 = eva.add(%0, %2) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %3 : !eva.cipher<4 x fixed_point>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.constant() {result_mod = 7 : si32, result_scale = 30 : si32, value = [1 : i32, 1 : i32, 1 : i32, 1 : i32]} : (none) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.multiply(%arg0, %arg0) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.multiply(%arg0, %0) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %3 = eva.add(%1, %2) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %3 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
