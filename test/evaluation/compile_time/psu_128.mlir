// RUN: fhe-tool --full-pass -mlir-timing -mlir-timing-display=list < %s | FileCheck %s
module  {
  func.func private @encryptedPSU(%a_id: tensor<128x8x!fhe.secret<i16>>, %a_data: tensor<128x!fhe.secret<i16>>,
                                  %b_id: tensor<128x8x!fhe.secret<i16>>, %b_data: tensor<128x!fhe.secret<i16>>) -> !fhe.secret<i16> {
    %c0_si16 = fhe.constant 0 : i16
    %c1_si16 = fhe.constant 1 : i16

    // First, sum all of A
    %1 = affine.for %i = 0 to 128 iter_args(%cur0 = %c0_si16) -> (!fhe.secret<i16>) {
        %1 = tensor.extract %a_data[%i] : tensor<128x!fhe.secret<i16>>
        %2 = fhe.add(%cur0, %1) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
         affine.yield %2 : !fhe.secret<i16>
    }

    // Now sum only those of B that aren't duplicates of A 
    %3 = affine.for %i = 0 to 128 iter_args(%cur1 = %1) -> (!fhe.secret<i16>) {
        // check if a[i] is a dupe of b[j]
        %4 = affine.for %j = 0 to 128 iter_args(%cur2 = %c1_si16) -> !fhe.secret<i16>{
            // compute a_id[i] == b_id[j]
            %5 =  affine.for %k = 0 to 8 iter_args(%cur3 = %c1_si16) -> !fhe.secret<i16>{
                %6 = tensor.extract %a_id[%i,%k] : tensor<128x8x!fhe.secret<i16>>
                %7 = tensor.extract %b_id[%j,%k] : tensor<128x8x!fhe.secret<i16>>
                %2 = fhe.sub(%6,%7) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                %9 = fhe.multiply(%2,%2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                %10 = fhe.sub(%c1_si16, %9) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                %11 = fhe.multiply(%10, %cur3) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
                affine.yield %11 : !fhe.secret<i16>
            }
            %12 = fhe.sub(%c1_si16, %5) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
            %13 = fhe.multiply(%12, %cur2) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
            affine.yield %13 : !fhe.secret<i16>
        }   
        %14 = tensor.extract %b_data[%i] : tensor<128x!fhe.secret<i16>>
        %15 = fhe.multiply(%14, %4) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
        %16 = fhe.add(%cur1, %15) : (!fhe.secret<i16>, !fhe.secret<i16>) -> !fhe.secret<i16>
        affine.yield %16 : !fhe.secret<i16>
    }
    return %3 : !fhe.secret<i16>
}
}
