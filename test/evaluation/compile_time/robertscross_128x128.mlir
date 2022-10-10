// RUN: fhe-tool --hir-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedRobertsCross(%img: tensor<16384x!fhe.secret<i16>>) -> tensor<16384x!fhe.secret<i16>> {
    %c16384 = arith.constant 16384 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c-1 =  arith.constant -1 : index

    // Each point p = img[x][y], where x is row and y is column, in the new image will equal:
    // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
    %r = affine.for %x = 0 to 128 iter_args(%imgx = %img) -> tensor<16384x!fhe.secret<i16>> {
      %1 = affine.for %y = 0 to 128 iter_args(%imgy = %imgx) -> tensor<16384x!fhe.secret<i16>> {

        // fetch img[x-1][y-1]
        %4 = arith.addi %x, %c-1 : index
        %5 = arith.muli %4, %c128 : index
        %6 = arith.addi %y, %c-1 : index
        %7 = arith.addi %5, %6 : index
        %8 = arith.remui %7, %c16384 : index
        %9 = tensor.extract %img[%8] : tensor<16384x!fhe.secret<i16>>

        // fetch img[x][y]
        %10 = arith.muli %x, %c128 : index
        %11 = arith.addi %10, %y : index
        %12 = arith.remui %11, %c16384 : index
        %13 = tensor.extract %img[%12] : tensor<16384x!fhe.secret<i16>>

        // subtract those two
        %14 = fhe.sub(%9,%13) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>

        // fetch img[x-1][y]
        %15 = arith.addi %x, %c-1 : index
        %16 = arith.muli %15, %c128 : index
        %17 = arith.addi %y, %c-1 : index
        %18 = arith.addi %16, %17 : index
        %19 = arith.remui %18, %c16384 : index
        %20 = tensor.extract %img[%19] : tensor<16384x!fhe.secret<i16>>

        // fetch img[x][y-1]
        %21 = arith.muli %x, %c128 : index
        %22 = arith.addi %y, %c-1 : index
        %23 = arith.addi %21, %22 : index
        %24 = arith.remui %23, %c16384 : index
        %25 = tensor.extract %img[%24] : tensor<16384x!fhe.secret<i16>>

        // subtract those two
        %26 = fhe.sub(%20,%25) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>

        // square each difference
        %27 = fhe.multiply(%14,%14) :  (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
        %28 = fhe.multiply(%26,%26) :  (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>

        // add the squares
        %29 = fhe.add(%27, %28) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>

        // save to result[x][y]
        %30 = tensor.insert %29 into %imgy[%12] : tensor<16384x!fhe.secret<i16>>
        affine.yield %30: tensor<16384x!fhe.secret<i16>>
      }
      affine.yield %1 : tensor<16384x!fhe.secret<i16>>
    }
    return %r : tensor<16384x!fhe.secret<i16>>
  }
}

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 5.1401 seconds

  ----Wall Time----  ----Name----
    5.1401 (100.0%)  root
    3.8575 ( 75.0%)  Canonicalizer
    0.5354 ( 10.4%)  CombineSimplifyPass
    0.2436 (  4.7%)  CSE
    0.1815 (  3.5%)  BatchingPass
    0.1451 (  2.8%)  UnrollLoopsPass
    0.1122 (  2.2%)  Tensor2BatchedSecretPass
    0.0626 (  1.2%)  NaryPass
    0.0004 (  0.0%)  Output
    0.0003 (  0.0%)  Parser
    0.0000 (  0.0%)  InternalOperandBatchingPass
    0.0000 (  0.0%)  (A) DominanceInfo
    0.0013 (  0.0%)  Rest
    5.1401 (100.0%)  Total