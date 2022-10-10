// RUN: fhe-tool -unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func.func private @encryptedPSU(%a_id: tensor<4x2x!fhe.secret<f64>>, %a_data: tensor<4x!fhe.secret<f64>>,
                                  %b_id: tensor<4x2x!fhe.secret<f64>>, %b_data: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c0_sf64 = fhe.constant 0.000000e+00 : f64
    %c1_sf64 = fhe.constant 1.0 : f64

    // First, sum all of A
    %1 = affine.for %i = 0 to 4 iter_args(%cur0 = %c0_sf64) -> (!fhe.secret<f64>) {
        %1 = tensor.extract %a_data[%i] : tensor<4x!fhe.secret<f64>>
        %2 = fhe.add(%cur0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
         affine.yield %2 : !fhe.secret<f64>
    }

    // Now sum only those of B that aren't duplicates of A 
    %3 = affine.for %i = 0 to 4 iter_args(%cur1 = %1) -> (!fhe.secret<f64>) {
        // check if a[i] is a dupe of b[j]
        %4 = affine.for %j = 0 to 4 iter_args(%cur2 = %c1_sf64) -> !fhe.secret<f64>{
            // compute a_id[i] == b_id[j]
            %5 =  affine.for %k = 0 to 2 iter_args(%cur3 = %c1_sf64) -> !fhe.secret<f64>{
                %6 = tensor.extract %a_id[%i,%k] : tensor<4x2x!fhe.secret<f64>>
                %7 = tensor.extract %b_id[%j,%k] : tensor<4x2x!fhe.secret<f64>>
                %2 = fhe.sub(%6,%7) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
                %9 = fhe.multiply(%2,%2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
                %10 = fhe.sub(%c1_sf64, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
                %11 = fhe.multiply(%10, %cur3) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
                affine.yield %11 : !fhe.secret<f64>
            }
            %12 = fhe.sub(%c1_sf64, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
            %13 = fhe.multiply(%12, %cur2) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
            affine.yield %13 : !fhe.secret<f64>
        }   
        %14 = tensor.extract %b_data[%i] : tensor<4x!fhe.secret<f64>>
        %15 = fhe.multiply(%14, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %16 = fhe.add(%cur1, %15) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        affine.yield %16 : !fhe.secret<f64>
    }
    return %3 : !fhe.secret<f64>
}
}
