//RUN:  heco --evawaterline --canonicalize < %s | FileCheck %s

module {
  func.func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
    %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %2 = eva.rotate(%1) by 2 : <4 x fixed_point>
    %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    %4 = eva.rotate(%3) by 3 : <4 x fixed_point>
    %5 = eva.add(%3, %4) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
    return %5 : !eva.cipher<4 x fixed_point>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point>, %arg1: !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point> {
//CHECK:     %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %2 = eva.rotate(%1) by 2 : <4 x fixed_point>
//CHECK:     %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     %4 = eva.rotate(%3) by 3 : <4 x fixed_point>
//CHECK:     %5 = eva.add(%3, %4) : (!eva.cipher<4 x fixed_point>, !eva.cipher<4 x fixed_point>) -> !eva.cipher<4 x fixed_point>
//CHECK:     return %5 : !eva.cipher<4 x fixed_point>
//CHECK:   }
//CHECK: }
