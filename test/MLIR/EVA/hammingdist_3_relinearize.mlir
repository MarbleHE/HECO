//RUN:  fhe-tool -relinearize --canonicalize < %s | FileCheck %s

module {
  func.func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point : 40 : -1>, %arg1: !eva.cipher<4 x fixed_point : 40 : -1>) -> !eva.cipher<4 x fixed_point : 40 : -1> {
    %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point : 40 : -1>, !eva.cipher<4 x fixed_point : 40 : -1>) -> !eva.cipher<4 x fixed_point : 40 : -1>
    %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point : 40 : -1>, !eva.cipher<4 x fixed_point : 40 : -1>) -> !eva.cipher<4 x fixed_point : 80 : -1>
    %2 = eva.rotate(%1) by 2 : <4 x fixed_point : 80 : -1>
    %3 = eva.add(%1, %2) : (!eva.cipher<4 x fixed_point : 80 : -1>, !eva.cipher<4 x fixed_point : 80 : -1>) -> !eva.cipher<4 x fixed_point : 80 : -1>
    %4 = eva.rotate(%3) by 3 : <4 x fixed_point : 80 : -1>
    %5 = eva.add(%3, %4) : (!eva.cipher<4 x fixed_point : 80 : -1>, !eva.cipher<4 x fixed_point : 80 : -1>) -> !eva.cipher<4 x fixed_point : 80 : -1>
    return %5 : !eva.cipher<4 x fixed_point : 80 : -1>
  }
}

//CHECK: module {
//CHECK:   func.func private @encryptedHammingDistance(%arg0: !eva.cipher<4 x fixed_point : 40 : -1>, %arg1: !eva.cipher<4 x fixed_point : 40 : -1>) -> !eva.cipher<4 x fixed_point : 40 : -1> {
//CHECK:     %0 = eva.sub(%arg0, %arg1) : (!eva.cipher<4 x fixed_point : 40 : -1>, !eva.cipher<4 x fixed_point : 40 : -1>) -> !eva.cipher<4 x fixed_point : 40 : -1>
//CHECK:     %1 = eva.multiply(%0, %0) : (!eva.cipher<4 x fixed_point : 40 : -1>, !eva.cipher<4 x fixed_point : 40 : -1>) -> !eva.cipher<4 x fixed_point : 80 : -1>
//CHECK:     %2 = eva.relinearize(%1) : !eva.cipher<4 x fixed_point : 80 : -1> -> !eva.cipher<4 x fixed_point : 80 : -1>
//CHECK:     %3 = eva.rotate(%2) by 2 : <4 x fixed_point : 80 : -1>
//CHECK:     %4 = eva.add(%2, %3) : (!eva.cipher<4 x fixed_point : 80 : -1>, !eva.cipher<4 x fixed_point : 80 : -1>) -> !eva.cipher<4 x fixed_point : 80 : -1>
//CHECK:     %5 = eva.rotate(%4) by 3 : <4 x fixed_point : 80 : -1>
//CHECK:     %6 = eva.add(%4, %5) : (!eva.cipher<4 x fixed_point : 80 : -1>, !eva.cipher<4 x fixed_point : 80 : -1>) -> !eva.cipher<4 x fixed_point : 80 : -1>
//CHECK:     return %6 : !eva.cipher<4 x fixed_point : 80 : -1>
//CHECK:   }
//CHECK: }
