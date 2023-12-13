//RUN:  heco --evametadata --canonicalize < %s | FileCheck %s

module {
  func.func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %2 = eva.relinearize(%1) : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %3 = eva.rotate(%2) by 2 : <4 x fixed_point>
    %4 = eva.add(%2, %3) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %5 = eva.rotate(%4) by 3 : <4 x fixed_point>
    %6 = eva.add(%4, %5) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %6 : !eva.cipher<4 x fixed_point>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.sub(%arg0, %arg1) {result_mod = 7 : si32, result_scale = 30 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.multiply(%0, %0) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.relinearize(%1) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %3 = eva.rotate(%2) by 2 {result_mod = 7 : si32, result_scale = 60 : si32} : <4 x fixed_point>
//CHECK:     %4 = eva.add(%2, %3) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %5 = eva.rotate(%4) by 3 {result_mod = 7 : si32, result_scale = 60 : si32} : <4 x fixed_point>
//CHECK:     %6 = eva.add(%4, %5) {result_mod = 7 : si32, result_scale = 60 : si32} : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %6 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
