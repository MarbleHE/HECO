// RUN: abc-opt -unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedRobertsCross(%img: tensor<64x!fhe.secret<f64>>) -> tensor<64x!fhe.secret<f64>> {
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %0 = affine.for %x = 0 to 8 iter_args(%imgx = %img) -> tensor<64x!fhe.secret<f64>> {
      %1 = affine.for %y = 0 to 8 iter_args(%imgy = %imgx) -> tensor<64x!fhe.secret<f64>> {
        %c0_sf64 = fhe.constant 0.000000e+00 : f64
        %2 = affine.for %j = -1 to 2 iter_args(%sumj = %c0_sf64) -> !fhe.secret<f64> {
          %3 = affine.for %i = -1 to 2 iter_args(%sumi = %sumj) -> !fhe.secret<f64> {
            %7 = arith.addi %x, %i : index
            %8 = arith.muli %7, %c8 : index
            %9 = arith.addi %y, %j : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %c64 : index
            %12 = tensor.extract %img[%11] : tensor<64x!fhe.secret<f64>>
            %13 = fhe.add(%sumi, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
            affine.yield %13 : !fhe.secret<f64>
          }
          affine.yield %3 : !fhe.secret<f64>
        }
        %3 = arith.muli %x, %c8 : index
        %4 = arith.addi %3, %y : index
        %5 = tensor.insert %2 into %imgy[%4] : tensor<64x!fhe.secret<f64>>
        affine.yield %5: tensor<64x!fhe.secret<f64>>
      }
      affine.yield %1 : tensor<64x!fhe.secret<f64>>
    }
    return %0 : tensor<64x!fhe.secret<f64>>
  }
}