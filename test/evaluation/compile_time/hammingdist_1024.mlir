// RUN: fhe-tool --hir-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedHammingDistance(%arg0: tensor<1024x!fhe.secret<i16>>, %arg1: tensor<1024x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_si16 = fhe.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 1024 iter_args(%arg6 = %c0_si16) -> (!fhe.secret<i16>) {
      %1 = tensor.extract %arg0[%arg2] : tensor<1024x!fhe.secret<i16>>
      %2 = tensor.extract %arg1[%arg2] : tensor<1024x!fhe.secret<i16>>
      %3 = fhe.sub(%1, %2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %4 = tensor.extract %arg0[%arg2] : tensor<1024x!fhe.secret<i16>>
      %5 = tensor.extract %arg1[%arg2] : tensor<1024x!fhe.secret<i16>>
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
  Total Execution Time: 0.1311 seconds

  ----Wall Time----  ----Name----
    0.1311 (100.0%)  root
    0.0702 ( 53.5%)  Canonicalizer
    0.0435 ( 33.2%)  NaryPass
    0.0055 (  4.2%)  CSE
    0.0041 (  3.2%)  BatchingPass
    0.0031 (  2.4%)  UnrollLoopsPass
    0.0030 (  2.3%)  Tensor2BatchedSecretPass
    0.0005 (  0.4%)  Output
    0.0004 (  0.3%)  InternalOperandBatchingPass
    0.0003 (  0.2%)  Parser
    0.0001 (  0.1%)  CombineSimplifyPass
    0.0000 (  0.0%)  (A) DominanceInfo
    0.0004 (  0.3%)  Rest
    0.1311 (100.0%)  Total