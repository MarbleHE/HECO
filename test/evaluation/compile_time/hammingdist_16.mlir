// RUN: fhe-tool --hir-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: tensor<16x!fhe.secret<i16>>, %arg1: tensor<16x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_si16 = fhe.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 16 iter_args(%arg6 = %c0_si16) -> (!fhe.secret<i16>) {
      %1 = tensor.extract %arg0[%arg2] : tensor<16x!fhe.secret<i16>>
      %2 = tensor.extract %arg1[%arg2] : tensor<16x!fhe.secret<i16>>
      %3 = fhe.sub(%1, %2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %4 = tensor.extract %arg0[%arg2] : tensor<16x!fhe.secret<i16>>
      %5 = tensor.extract %arg1[%arg2] : tensor<16x!fhe.secret<i16>>
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
  Total Execution Time: 0.0026 seconds

  ----Wall Time----  ----Name----
    0.0026 (100.0%)  root
    0.0011 ( 41.9%)  Canonicalizer
    0.0004 ( 17.3%)  Output
    0.0003 ( 10.0%)  Parser
    0.0001 (  4.2%)  CSE
    0.0001 (  3.9%)  UnrollLoopsPass
    0.0001 (  3.2%)  Tensor2BatchedSecretPass
    0.0001 (  2.5%)  BatchingPass
    0.0000 (  1.0%)  NaryPass
    0.0000 (  0.6%)  InternalOperandBatchingPass
    0.0000 (  0.1%)  CombineSimplifyPass
    0.0000 (  0.1%)  (A) DominanceInfo
    0.0004 ( 15.4%)  Rest
    0.0026 (100.0%)  Total