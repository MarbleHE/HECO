//RUN:  heco --evametadata < %s | FileCheck %s

module {
  func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.multiply(%arg0, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.relinearize(%0) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %2 = eva.multiply(%1, %arg0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %3 = eva.relinearize(%2) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %4 = eva.rescale(%3) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %5 = eva.multiply(%arg1, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %6 = eva.relinearize(%5) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %7 = eva.modswitch(%6) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %8 = eva.multiply(%4, %7) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %9 = eva.relinearize(%8) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %10 = eva.rescale(%9) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %10 : !eva.cipher<4 x fixed_point>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedPoly(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.multiply(%arg0, %arg0) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.relinearize(%0) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.multiply(%1, %arg0) {result_mod = 7 : si32, result_scale = 90 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %3 = eva.relinearize(%2) {result_mod = 7 : si32, result_scale = 90 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %4 = eva.rescale(%3) {result_mod = 6 : si32, result_scale = 30 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %5 = eva.multiply(%arg1, %arg1) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %6 = eva.relinearize(%5) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %7 = eva.modswitch(%6) {result_mod = 6 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %8 = eva.multiply(%4, %7) {result_mod = 6 : si32, result_scale = 90 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %9 = eva.relinearize(%8) {result_mod = 6 : si32, result_scale = 90 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %10 = eva.rescale(%9) {result_mod = 5 : si32, result_scale = 30 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %10 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
