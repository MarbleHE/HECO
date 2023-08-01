module  {
  func.func private @encryptedDotProduct_8(%arg0: tensor<8x!fhe.secret<i16>>, %arg1: tensor<8x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0 = arith.constant 0 : index
    %c0_si16 = fhe.constant 0 : i16
    %0 = affine.for %arg2 = 0 to 4 iter_args(%iter = %c0_si16) -> (!fhe.secret<i16>) {
      %1 = tensor.extract %arg0[%arg2] : tensor<8x!fhe.secret<i16>>
      %2 = tensor.extract %arg1[%arg2] : tensor<8x!fhe.secret<i16>>
      %3 = fhe.multiply(%1, %2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>    
      %4 = fhe.add(%iter, %3) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      affine.yield %4 : !fhe.secret<i16>
    }
    return %0 : !fhe.secret<i16>
  }
}