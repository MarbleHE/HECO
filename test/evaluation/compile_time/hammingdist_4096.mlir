// RUN: fhe-tool --hir-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: tensor<4096x!fhe.secret<i16>>, %arg1: tensor<4096x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_si16 = fhe.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 4096 iter_args(%arg6 = %c0_si16) -> (!fhe.secret<i16>) {
      %1 = tensor.extract %arg0[%arg2] : tensor<4096x!fhe.secret<i16>>
      %2 = tensor.extract %arg1[%arg2] : tensor<4096x!fhe.secret<i16>>
      %3 = fhe.sub(%1, %2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %4 = tensor.extract %arg0[%arg2] : tensor<4096x!fhe.secret<i16>>
      %5 = tensor.extract %arg1[%arg2] : tensor<4096x!fhe.secret<i16>>
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
  Total Execution Time: 1.2922 seconds

  ----Wall Time----  ----Name----
    1.2922 (100.0%)  root
    0.7834 ( 60.6%)  NaryPass
    0.4291 ( 33.2%)  Canonicalizer
    0.0240 (  1.9%)  CSE
    0.0216 (  1.7%)  BatchingPass
    0.0138 (  1.1%)  UnrollLoopsPass
    0.0136 (  1.1%)  Tensor2BatchedSecretPass
    0.0044 (  0.3%)  InternalOperandBatchingPass
    0.0010 (  0.1%)  Output
    0.0004 (  0.0%)  CombineSimplifyPass
    0.0002 (  0.0%)  Parser
    0.0000 (  0.0%)  (A) DominanceInfo
    0.0006 (  0.0%)  Rest
    1.2922 (100.0%)  Total