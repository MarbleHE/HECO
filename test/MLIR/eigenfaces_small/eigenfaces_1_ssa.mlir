// RUN: fhe-tool -unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func.func private @encryptedEigenfaces(%x: tensor<4x!fhe.secret<f64>>, %d: tensor<4x4x!fhe.secret<f64>>) -> tensor<4x!fhe.secret<f64>> {
    %c0_sf64 = fhe.constant 0.000000e+00 : f64
    %0 = linalg.init_tensor [4]: tensor<4x!fhe.secret<f64>> //TODO: replace with tensor.empty() :  tensor<4x!fhe.secret<f64>> after next MLIR rebase!
    %1 = affine.for %i = 0 to 4 iter_args(%cur0 = %0) -> (tensor<4x!fhe.secret<f64>>) {
      // %cur0 is the current result vector. 
      %2 = affine.for %j = 0 to 4 iter_args(%cur1 = %c0_sf64) -> !fhe.secret<f64>{
        // %2 is the running sum of the current euclidean distance
        %1 = tensor.extract %d[%i,%j] : tensor<4x4x!fhe.secret<f64>>
        %2 = fhe.multiply(%1,%1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %3 = tensor.extract %x[%j] : tensor<4x!fhe.secret<f64>>
        %4 = fhe.multiply(%3, %3) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %5 = fhe.sub(%2, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>         
        %6 = fhe.add(%cur1, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        affine.yield %6 : !fhe.secret<f64>
      }
      %7 = tensor.insert %2 into %cur0[%i] :  tensor<4x!fhe.secret<f64>>
      affine.yield %7 : tensor<4x!fhe.secret<f64>>
    }
    return %1 : tensor<4x!fhe.secret<f64>>
  }
}

//TODO: ...this magically breaks with i16 and muli instead of float??