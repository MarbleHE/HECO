// RUN: abc-opt -unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func private @encryptedRobertsCross(%img: tensor<64x!fhe.secret<f64>>) -> tensor<64x!fhe.secret<f64>> {
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c-1 =  arith.constant -1 : index

    // first kernel
    %w1 = tensor.from_elements %c0, %c0, %c1, %c0, %c-1, %c0, %c0, %c0, %c0 :  tensor<3x3xindex>
    %k1 = affine.for %x = 0 to 8 iter_args(%imgx = %img) -> tensor<64x!fhe.secret<f64>> {
      %1 = affine.for %y = 0 to 8 iter_args(%imgy = %imgx) -> tensor<64x!fhe.secret<f64>> {
        %c0_sf64 = fhe.constant 0.000000e+00 : f64
        %2 = affine.for %j = -1 to 2 iter_args(%sumj = %c0_sf64) -> !fhe.secret<f64> {
          %3 = affine.for %i = -1 to 2 iter_args(%sumi = %sumj) -> !fhe.secret<f64> {
            %4 = arith.addi %i, %c1 : index
            %5 = arith.addi %j, %c1 : index
            %6 = tensor.extract %w1[%4,%5] : tensor<3x3xindex>
            %7 = arith.addi %x, %i : index
            %8 = arith.muli %7, %c8 : index
            %9 = arith.addi %y, %j : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %c64 : index
            %12 = tensor.extract %img[%11] : tensor<64x!fhe.secret<f64>>
            %13 = fhe.multiply(%12,%6) :  (!fhe.secret<f64>, index) -> !fhe.secret<f64>
            %14 = fhe.add(%sumi, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
            affine.yield %14 : !fhe.secret<f64>
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

    // second kernel
    %w2 = tensor.from_elements %c0, %c-1, %c0, %c0, %c0, %c-1, %c0, %c0, %c0 :  tensor<3x3xindex>
    %k2 = affine.for %x = 0 to 8 iter_args(%imgx = %img) -> tensor<64x!fhe.secret<f64>> {
      %1 = affine.for %y = 0 to 8 iter_args(%imgy = %imgx) -> tensor<64x!fhe.secret<f64>> {
        %c0_sf64 = fhe.constant 0.000000e+00 : f64
        %2 = affine.for %j = -1 to 2 iter_args(%sumj = %c0_sf64) -> !fhe.secret<f64> {
          %3 = affine.for %i = -1 to 2 iter_args(%sumi = %sumj) -> !fhe.secret<f64> {
            %4 = arith.addi %i, %c1 : index
            %5 = arith.addi %j, %c1 : index
            %6 = tensor.extract %w2[%4,%5] : tensor<3x3xindex>
            %7 = arith.addi %x, %i : index
            %8 = arith.muli %7, %c8 : index
            %9 = arith.addi %y, %j : index
            %10 = arith.addi %8, %9 : index
            %11 = arith.remui %10, %c64 : index
            %12 = tensor.extract %img[%11] : tensor<64x!fhe.secret<f64>>
            %13 = fhe.multiply(%12,%6) :  (!fhe.secret<f64>, index) -> !fhe.secret<f64>
            %14 = fhe.add(%sumi, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
            affine.yield %14 : !fhe.secret<f64>
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

    // compute x^2 + y^2
    %r = affine.for %x = 0 to 8 iter_args(%imgx = %img) -> tensor<64x!fhe.secret<f64>> {
      %1 = affine.for %y = 0 to 8 iter_args(%imgy = %imgx) -> tensor<64x!fhe.secret<f64>> {
        %2 = arith.muli %x, %c8 : index
        %3 = arith.addi %2, %y : index
        %4 = tensor.extract %k1[%3] : tensor<64x!fhe.secret<f64>>
        %5 = tensor.extract %k2[%3] : tensor<64x!fhe.secret<f64>>
        %6 = fhe.multiply(%4,%4) :  (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %7 = fhe.multiply(%5,%5) :  (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %8 = fhe.add(%6,%7) :  (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %9 = tensor.insert %8 into %imgy[%3] : tensor<64x!fhe.secret<f64>>
        affine.yield %9 : tensor<64x!fhe.secret<f64>>
      }
      affine.yield %1 : tensor<64x!fhe.secret<f64>>
    }
    return %r : tensor<64x!fhe.secret<f64>>
  }
}