// RUN: fhe-tool --hir-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: tensor<64x!fhe.secret<i16>>, %arg1: tensor<64x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_si16 = fhe.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 64 iter_args(%arg6 = %c0_si16) -> (!fhe.secret<i16>) {
      %1 = tensor.extract %arg0[%arg2] : tensor<64x!fhe.secret<i16>>
      %2 = tensor.extract %arg1[%arg2] : tensor<64x!fhe.secret<i16>>
      %3 = fhe.sub(%1, %2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %4 = tensor.extract %arg0[%arg2] : tensor<64x!fhe.secret<i16>>
      %5 = tensor.extract %arg1[%arg2] : tensor<64x!fhe.secret<i16>>
      %6 = fhe.sub(%4, %5) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %7 = fhe.multiply(%3, %6) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %8 = fhe.add(%arg6, %7) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      affine.yield %8 : !fhe.secret<i16>
    }
    return %0 : !fhe.secret<i16>
  }
}

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0059 seconds

  ----Wall Time----  ----Name----
    0.0059 (100.0%)  root
    0.0036 ( 60.5%)  Canonicalizer
    0.0005 (  8.6%)  Output
    0.0004 (  6.0%)  CSE
    0.0003 (  4.5%)  Parser
    0.0002 (  3.8%)  BatchingPass
    0.0002 (  3.8%)  UnrollLoopsPass
    0.0002 (  3.4%)  Tensor2BatchedSecretPass
    0.0001 (  2.0%)  NaryPass
    0.0000 (  0.4%)  InternalOperandBatchingPass
    0.0000 (  0.2%)  CombineSimplifyPass
    0.0000 (  0.0%)  (A) DominanceInfo
    0.0004 (  6.8%)  Rest
    0.0059 (100.0%)  Total