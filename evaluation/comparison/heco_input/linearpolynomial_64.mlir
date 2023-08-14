module  {
  func.func private @encryptedLinearPolynomial_64(%a: tensor<64x!fhe.secret<i16>>, %b: tensor<64x!fhe.secret<i16>>, %x: tensor<64x!fhe.secret<i16>>, %y: tensor<64x!fhe.secret<i16>>) -> tensor<64x!fhe.secret<i16>> {
    %0 = affine.for %i = 0 to 64 iter_args(%iter = %y) -> (tensor<64x!fhe.secret<i16>>) {
      %ai = tensor.extract %a[%i] : tensor<64x!fhe.secret<i16>>
      %bi = tensor.extract %b[%i] : tensor<64x!fhe.secret<i16>>
      %xi = tensor.extract %x[%i] : tensor<64x!fhe.secret<i16>>
      %yi = tensor.extract %y[%i] : tensor<64x!fhe.secret<i16>>
      %axi = fhe.multiply(%ai, %xi) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %t1 = fhe.sub(%yi, %axi) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %t2 = fhe.sub(%t1, %bi) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
      %out = tensor.insert %t2 into %iter[%i] : tensor<64x!fhe.secret<i16>>
      affine.yield %out : tensor<64x!fhe.secret<i16>>
    }
    return %0 : tensor<64x!fhe.secret<i16>>
  }
}