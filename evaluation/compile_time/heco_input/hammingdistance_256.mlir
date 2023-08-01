// RUN: heco --full-pass -mlir-timing -mlir-timing-display=list < %s
module  {
  func.func private @encryptedHammingDistance_256(%arg0: tensor<256x!fhe.secret<i16>>, %arg1: tensor<256x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_si16 = fhe.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 256 iter_args(%arg6 = %c0_si16) -> (!fhe.secret<i16>) {
      %1 = tensor.extract %arg0[%arg2] : tensor<256x!fhe.secret<i16>>
      %2 = tensor.extract %arg1[%arg2] : tensor<256x!fhe.secret<i16>>
      %3 = fhe.sub(%1, %2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %4 = tensor.extract %arg0[%arg2] : tensor<256x!fhe.secret<i16>>
      %5 = tensor.extract %arg1[%arg2] : tensor<256x!fhe.secret<i16>>
      %6 = fhe.sub(%4, %5) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %7 = fhe.multiply(%3, %6) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %8 = fhe.add(%arg6, %7) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      affine.yield %8 : !fhe.secret<i16>
    }
    return %0 : !fhe.secret<i16>
  }
}