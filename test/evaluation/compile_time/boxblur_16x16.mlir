// RUN: fhe-tool --hir-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedBoxBlur(%arg0: tensor<256x!fhe.secret<i16>>) -> tensor<256x!fhe.secret<i16>> {
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %0:4 = affine.for %arg1 = 0 to 16 iter_args(%arg2 = %c16, %arg3 = %c256, %arg4 = %arg0, %arg5 = %arg0) -> (index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>) {
      %1:5 = affine.for %arg6 = 0 to 16 iter_args(%arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg1, %arg10 = %arg4, %arg11 = %arg5) -> (index, index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>) {
        %c0_si16 = fhe.constant 0 : i16
        %2:7 = affine.for %arg12 = -1 to 2 iter_args(%arg13 = %arg7, %arg14 = %arg8, %arg15 = %arg6, %arg16 = %arg9, %arg17 = %arg10, %arg18 = %arg11, %arg19 = %c0_si16) -> (index, index, index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>, !fhe.secret<i16>) {
          %6:8 = affine.for %arg20 = -1 to 2 iter_args(%arg21 = %arg13, %arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg12, %arg25 = %arg16, %arg26 = %arg17, %arg27 = %arg18, %arg28 = %arg19) -> (index, index, index, index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>, !fhe.secret<i16>) {
            %7 = arith.addi %arg25, %arg20 : index
            %8 = arith.muli %7, %arg21 : index
            %9 = arith.addi %arg23, %arg24 : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %arg22 : index
            %12 = tensor.extract %arg27[%11] : tensor<256x!fhe.secret<i16>>
            %13 = fhe.add(%arg28, %12) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
            affine.yield %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %13 : index, index, index, index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>, !fhe.secret<i16>
          }
          affine.yield %6#0, %6#1, %6#2, %6#4, %6#5, %6#6, %6#7 : index, index, index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>, !fhe.secret<i16>
        }
        %3 = arith.muli %2#0, %2#3 : index
        %4 = arith.addi %3, %2#2 : index
        %5 = arith.remui %4, %c256 : index
        %6 = tensor.insert %2#6 into %2#4[%5] : tensor<256x!fhe.secret<i16>>
        affine.yield %2#0, %2#1, %2#3, %6, %2#5 : index, index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>
      }
      affine.yield %1#0, %1#1, %1#3, %1#4 : index, index, tensor<256x!fhe.secret<i16>>, tensor<256x!fhe.secret<i16>>
    }
    return %0#2 : tensor<256x!fhe.secret<i16>>
  }
}