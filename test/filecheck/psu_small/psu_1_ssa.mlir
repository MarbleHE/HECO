// RUN: heco --unroll-loops --canonicalize --cse < %s | FileCheck %s
module  {
  func.func private @encryptedPSU(%a_id: tensor<4x2x!fhe.secret<i16>>, %a_data: tensor<4x!fhe.secret<i16>>,
                                  %b_id: tensor<4x2x!fhe.secret<i16>>, %b_data: tensor<4x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0_si16 = fhe.constant 0 : i16
    %c1_si16 = fhe.constant 1 : i16

    // First, sum all of A
    %sum_a = affine.for %i = 0 to 4 iter_args(%cur0 = %c0_si16) -> (!fhe.secret<i16>) {
        %1 = tensor.extract %a_data[%i] : tensor<4x!fhe.secret<i16>>
        %2 = fhe.add(%cur0, %1) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
         affine.yield %2 : !fhe.secret<i16>
    }

    // Now sum only those of B that aren't duplicates of A 
    %sum = affine.for %i = 0 to 4 iter_args(%cur_sum = %sum_a) -> (!fhe.secret<i16>) {
        // check if a[i] is a dupe of b[j]
        %unique = affine.for %j = 0 to 4 iter_args(%cur_unique = %c1_si16) -> !fhe.secret<i16>{
            // compute a_id[i] == b_id[j]
            %equal =  affine.for %k = 0 to 2 iter_args(%cur_equal = %c1_si16) -> !fhe.secret<i16>{
                %6 = tensor.extract %a_id[%i,%k] : tensor<4x2x!fhe.secret<i16>>
                %7 = tensor.extract %b_id[%j,%k] : tensor<4x2x!fhe.secret<i16>>
                %8 = fhe.sub(%6,%7) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                %9 = fhe.multiply(%8,%8) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                %10 = fhe.sub(%c1_si16, %9) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                %11 = fhe.multiply(%10, %cur_equal) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                affine.yield %11 : !fhe.secret<i16>
            }
            %nequal = fhe.sub(%c1_si16, %equal) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
            %13 = fhe.multiply(%cur_unique, %nequal) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
            affine.yield %13 : !fhe.secret<i16>
        }   
        %14 = tensor.extract %b_data[%i] : tensor<4x!fhe.secret<i16>>
        %15 = fhe.multiply(%14, %unique) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
        %16 = fhe.add(%cur_sum, %15) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
        affine.yield %16 : !fhe.secret<i16>
    }
    return %sum : !fhe.secret<i16>
}
}


// Original algorithm uses:
// n rotations for sum A
// 2*n*n*k rotations for all the xor extracts
// 2*n*n*k mults for all the equal
// n*n mults for the uniques
// n rotations for the b[i] extracts
// n multiplies for the unique * b[i]
// ______________________________
//   2*n*n*k + n rotations 
// + 2*n*n*k + n*n + n mults 
// => for n=128, k = 8, that is 278656 mults and 262272 rotations
// assuming 10ms for each, it should take 1.5h