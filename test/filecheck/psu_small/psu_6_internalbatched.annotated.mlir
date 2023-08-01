module {
  func.func private @encryptedPSU(%a_id: !fhe.batched_secret<8 x i16>, %a_data: !fhe.batched_secret<4 x i16>, %b_id: !fhe.batched_secret<8 x i16>, %b_data: !fhe.batched_secret<4 x i16>) -> !fhe.secret<i16> {
    %cst = fhe.constant dense<1> : tensor<8xi16>
    %a_data_resized = fhe.materialize(%a_data) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>
    %b_data_resized = fhe.materialize(%b_data) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>

     // SUM ALL OF A: O(log(n)) instead of O(n)
    %52 = fhe.rotate(%a_data_resized) by 2 : <8 x i16>
    %53 = fhe.add(%a_data_resized, %52) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %54 = fhe.rotate(%53) by 1 : <8 x i16>
    %sum_a = fhe.add(%53, %54) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>


  // NOW WE HAVE O(n) nequal computations  instead of O(n^2)
  // Each of them uses O(1) instead of O(k) to compute the xor
  // Each of them uses O(log(k)) instead of O(k) mults to compute equal

    // compute a_id[i] != b_id[i] for each iteration
    // Note that the rotation is by 2 x the offset because of the column-major encoding! 
    // --> used for (0,0), (1,1), (2,2), (3,3)
    %1 = fhe.sub(%a_id, %b_id) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %2 = fhe.multiply(%1, %1) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %3 = fhe.sub(%cst, %2) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    // update nequal: multiply all bits, then negate
    %4 = fhe.rotate(%3) by 1 : <8 x i16>
    %5 = fhe.multiply(%3, %4) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %6 = fhe.sub(%cst, %5) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    // compute a_id[i] != b_id[i-1 % 4] for each iteration
    // Note that the rotation is by 2 x the offset because of the column-major encoding! 
    // --> used for (0,3), (1,0), (2,1), (3,2)
    // compute xor
    %7 = fhe.rotate(%b_id) by -2 : <8 x i16>
    %8 = fhe.sub(%a_id, %7) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %9 = fhe.multiply(%8, %8) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %10 = fhe.sub(%cst, %9) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    // update nequal: multiply all bits, then negate
    %11 = fhe.rotate(%10) by 1 : <8 x i16>
    %12 = fhe.multiply(%10, %11) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %13 = fhe.sub(%cst, %12) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
   
    // compute a_id[i] != b_id[i-2 % 4] for each iteration
    // Note that the rotation is by 2 x the offset because of the column-major encoding! 
    // --> used for (0,2), (1,3), (2,0), (3,1)
    %14 = fhe.rotate(%b_id) by -4 : <8 x i16>
    %15 = fhe.sub(%a_id, %14) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %16 = fhe.multiply(%15, %15) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %17 = fhe.sub(%cst, %16) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    // update nequal: multiply all bits, then negate
    %18 = fhe.rotate(%17) by 1 : <8 x i16>
    %19 = fhe.multiply(%17, %18) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %20 = fhe.sub(%cst, %19) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    // compute a_id[i] != b_id[i-3 % 4] for each iteration
    // Note that the rotation is by 2 x the offset because of the column-major encoding! 
    // --> used for (0,1), (1,2), (2,3), (3,0)
    %21 = fhe.rotate(%b_id) by -6 : <8 x i16>
    %22 = fhe.sub(%a_id, %21) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %23 = fhe.multiply(%22, %22) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %24 = fhe.sub(%cst, %23) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    // update nequal: multiply all bits, then negate
    %25 = fhe.rotate(%24) by 1 : <8 x i16>
    %26 = fhe.multiply(%24, %25) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %27 = fhe.sub(%cst, %26) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    

// Now we compute O(n) unique * b[i], each sadly using O(n) rotates rather than the ideal O(1) 
// This also uses O(n*n) multiplications, since  
    %29 = fhe.rotate(%27) by -1 : <8 x i16>
    %30 = fhe.rotate(%20) by -1 : <8 x i16>
    %31 = fhe.rotate(%6) by -1 : <8 x i16>
    %32 = fhe.rotate(%13) by -1 : <8 x i16>
     // unique * b[0]
    %33 = fhe.multiply(%b_data_resized, %29, %30, %31, %32) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %34 = fhe.rotate(%20) by -2 : <8 x i16>
    %35 = fhe.rotate(%13) by -2 : <8 x i16>
    %36 = fhe.rotate(%27) by -2 : <8 x i16>
    %37 = fhe.rotate(%6) by -2 : <8 x i16>
    // unique * b[1]
    %38 = fhe.multiply(%b_data_resized, %34, %35, %36, %37) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %39 = fhe.rotate(%13) by -3 : <8 x i16>
    %40 = fhe.rotate(%6) by -3 : <8 x i16>
    %41 = fhe.rotate(%20) by -3 : <8 x i16>
    %42 = fhe.rotate(%27) by -3 : <8 x i16>
    // unique * b[2]
    %43 = fhe.multiply(%b_data_resized, %39, %40, %41, %42) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %44 = fhe.rotate(%6) by -4 : <8 x i16>
    %45 = fhe.rotate(%27) by -4 : <8 x i16>
    %46 = fhe.rotate(%13) by -4 : <8 x i16>
    %47 = fhe.rotate(%20) by -4 : <8 x i16>
    // unique * b[3]
    %48 = fhe.multiply(%b_data_resized, %44, %45, %46, %47) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    

// the final sum uses O(log n) rotations
    // running sum of loop (unique * b[i]) + sum of a[i] from before
    %49 = fhe.rotate(%48) by 5 : <8 x i16>
    %50 = fhe.rotate(%43) by 6 : <8 x i16>
    %51 = fhe.rotate(%38) by 7 : <8 x i16>  
    %56 = fhe.rotate(%sum_a) by 3 : <8 x i16>
    %57 = fhe.add(%49, %50, %51, %33, %56) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %58 = fhe.extract %57[0] : <8 x i16>
    return %58 : !fhe.secret<i16>
  }
}

// batched algorithm uses:
// log n rotations for sum A
// 0 rotations for xor extracts :) 
// n mults for the squares
// n * log(k) mults for the equal 
// n * log(k) rotations for the equal
// n * n rotations for the unique * b[i] alignment
// n * n mults for the unique * b[i] alignment
// log n rotates for the final sum
// ______________________________
//   n*n + n * log(k)  + 2 * log n rotations
// + n*n + n * log(k) + n mults
// => for n = 128, k = 8, that is 16896 mults and 16775 rotatons.
// assumuing 10ms for each, it should take 5.6 min
// 16x speedup over the naive approach!