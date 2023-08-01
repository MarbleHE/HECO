// RUN: heco --unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func.func private @encryptedBoxBlur_64x64(%arg0: tensor<4096x!fhe.secret<i16>>) -> tensor<4096x!fhe.secret<i16>> {
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %0 = affine.for %x = 0 to 64 iter_args(%arg0_x = %arg0) -> (tensor<4096x!fhe.secret<i16>>) {
      %1 = affine.for %y = 0 to 64 iter_args(%arg0_y = %arg0_x) -> (tensor<4096x!fhe.secret<i16>>) {
        %c0_si16 = fhe.constant 0 : i16
        %2 = affine.for %j = -1 to 2 iter_args(%value_j = %c0_si16) -> (!fhe.secret<i16>) {
          %6 = affine.for %i = -1 to 2 iter_args(%value_i = %value_j) -> (!fhe.secret<i16>) {
            %7 = arith.addi %x, %i : index
            %8 = arith.muli %7, %c64 : index
            %9 = arith.addi %y, %j : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %c4096 : index
            %12 = tensor.extract %arg0[%11] : tensor<4096x!fhe.secret<i16>>
            %13 = fhe.add(%value_i, %12) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
            affine.yield %13 : !fhe.secret<i16>
          }
          affine.yield %6 : !fhe.secret<i16>
        }
        %3 = arith.muli %c64, %x : index
        %4 = arith.addi %3, %y : index
        %5 = arith.remui %4, %c4096 : index
        %6 = tensor.insert %2 into %arg0_y[%5] : tensor<4096x!fhe.secret<i16>>
        affine.yield %6 : tensor<4096x!fhe.secret<i16>>
      }
      affine.yield %1 : tensor<4096x!fhe.secret<i16>>
    }
    return %0 : tensor<4096x!fhe.secret<i16>>
  }
}