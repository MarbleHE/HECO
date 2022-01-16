builtin.module  {
  func private @encryptedHammingDistance(%arg0: !fhe.batched_secret<f64>, %arg1: !fhe.batched_secret<f64>) -> !fhe.secret<f64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %2 = fhe.sub(%arg0, %arg1) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %3 = fhe.mul(%2, %2) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %4 = fhe.rotate %3, %c2 : !fhe.batched_secret<f64>, index, !fhe.batched_secret<f64>
    %5 = fhe.add(%4,%3) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %6 = fhe.rotate %5, %c1 : !fhe.batched_secret<f64>, index, !fhe.batched_secret<f64>
    %7 = fhe.add(%6,%5) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %8 = fhe.extract %7[c0] : !fhe.secret<f64> //actually a no-op, because a scalar is just "value at slot 0"
    return %8 : !fhe.secret<f64>
  }
}

