module {
  func.func private @encryptedPSU(%a_id: !fhe.batched_secret<8 x i16>, %a_data: !fhe.batched_secret<4 x i16>, %b_id: !fhe.batched_secret<8 x i16>, %b_data: !fhe.batched_secret<4 x i16>) -> !fhe.secret<i16> {
    %cst = fhe.constant dense<1> : tensor<8xi16>
    %a_data_resized = fhe.materialize(%a_data) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>
    %b_data_resized = fhe.materialize(%b_data) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>

     // SUM ALL OF A
    %52 = fhe.rotate(%a_data_resized) by 2 : <8 x i16>
    %53 = fhe.add(%a_data_resized, %52) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %54 = fhe.rotate(%53) by 1 : <8 x i16>
    %sum_a = fhe.add(%53, %54) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>

    // compute a_id[i] != b_id[i] for each iteration
    // --> used for (0,0), (1,1), (2,2), (3,3)
    %1 = fhe.sub(%a_id, %b_id) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %2 = fhe.multiply(%1, %1) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %3 = fhe.sub(%cst, %2) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    // update unique??
    %4 = fhe.rotate(%3) by 1 : <8 x i16>
    %5 = fhe.multiply(%3, %4) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %6 = fhe.sub(%cst, %5) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %7 = fhe.rotate(%b_id) by 6 : <8 x i16>
    %8 = fhe.sub(%a_id, %7) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %9 = fhe.multiply(%8, %8) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %10 = fhe.sub(%cst, %9) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %11 = fhe.rotate(%10) by 1 : <8 x i16>
    %12 = fhe.multiply(%10, %11) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %13 = fhe.sub(%cst, %12) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %14 = fhe.rotate(%b_id) by 4 : <8 x i16>
    %15 = fhe.sub(%a_id, %14) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %16 = fhe.multiply(%15, %15) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %17 = fhe.sub(%cst, %16) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %18 = fhe.rotate(%17) by 1 : <8 x i16>
    %19 = fhe.multiply(%17, %18) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %20 = fhe.sub(%cst, %19) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %21 = fhe.rotate(%b_id) by 2 : <8 x i16>
    %22 = fhe.sub(%a_id, %21) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %23 = fhe.multiply(%22, %22) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %24 = fhe.sub(%cst, %23) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %25 = fhe.rotate(%24) by 1 : <8 x i16>
    %26 = fhe.multiply(%24, %25) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %27 = fhe.sub(%cst, %26) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %29 = fhe.rotate(%27) by 7 : <8 x i16>
    %30 = fhe.rotate(%20) by 7 : <8 x i16>
    %31 = fhe.rotate(%6) by 7 : <8 x i16>
    %32 = fhe.rotate(%13) by 7 : <8 x i16>
    %33 = fhe.multiply(%b_data_resized, %29, %30, %31, %32) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %34 = fhe.rotate(%20) by 6 : <8 x i16>
    %35 = fhe.rotate(%13) by 6 : <8 x i16>
    %36 = fhe.rotate(%27) by 6 : <8 x i16>
    %37 = fhe.rotate(%6) by 6 : <8 x i16>
    %38 = fhe.multiply(%b_data_resized, %34, %35, %36, %37) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %39 = fhe.rotate(%13) by 5 : <8 x i16>
    %40 = fhe.rotate(%6) by 5 : <8 x i16>
    %41 = fhe.rotate(%20) by 5 : <8 x i16>
    %42 = fhe.rotate(%27) by 5 : <8 x i16>
    %43 = fhe.multiply(%b_data_resized, %39, %40, %41, %42) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %44 = fhe.rotate(%6) by 4 : <8 x i16>
    %45 = fhe.rotate(%27) by 4 : <8 x i16>
    %46 = fhe.rotate(%13) by 4 : <8 x i16>
    %47 = fhe.rotate(%20) by 4 : <8 x i16>
    %48 = fhe.multiply(%b_data_resized, %44, %45, %46, %47) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    

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



module {
  func.func private @encryptedPSU(%a_id: !fhe.batched_secret<8 x i16>, %a_data: !fhe.batched_secret<4 x i16>, %b_id: !fhe.batched_secret<8 x i16>, %b_data: !fhe.batched_secret<4 x i16>) -> !fhe.secret<i16> {
    %cst = fhe.constant dense<1> : tensor<8xi16>
    %a_data_resized = fhe.materialize(%a_data) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>
    %b_data_resized = fhe.materialize(%b_data) : (!fhe.batched_secret<4 x i16>) -> !fhe.batched_secret<8 x i16>
    

    
    %52 = fhe.rotate(%a_data_resized) by 2 : <8 x i16>
    %53 = fhe.add(%a_data_resized, %52) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %54 = fhe.rotate(%53) by 1 : <8 x i16>
    %sum_a = fhe.add(%53, %54) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %56 = fhe.rotate(%sum_a) by 3 : <8 x i16>
    
    // compute a_id[i] != b_id[i] for each iteration
    // --> used for (0,0), (1,1), (2,2), (3,3)
    %1 = fhe.sub(%a_id, %b_id) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>  
    %2 = fhe.multiply(%1, %1) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %3 = fhe.sub(%cst, %2) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %4 = fhe.rotate(%3) by 1 : <8 x i16>
    %5 = fhe.multiply(%3, %4) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %6 = fhe.sub(%cst, %5) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    // compute a_id[i] != b_id[i-2 % n] for each operation
    %7 = fhe.rotate(%b_id) by -2 : <8 x i16>
    %8 = fhe.sub(%a_id, %7) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %9 = fhe.multiply(%8, %8) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %10 = fhe.sub(%cst, %9) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %11 = fhe.rotate(%10) by 1 : <8 x i16>
    %12 = fhe.multiply(%10, %11) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %13 = fhe.sub(%cst, %12) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %14 = fhe.rotate(%b_id) by -4 : <8 x i16>
    %15 = fhe.sub(%a_id, %14) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %16 = fhe.multiply(%15, %15) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %17 = fhe.sub(%cst, %16) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %18 = fhe.rotate(%17) by 1 : <8 x i16>
    %19 = fhe.multiply(%17, %18) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %20 = fhe.sub(%cst, %19) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %21 = fhe.rotate(%b_id) by -6 : <8 x i16>
    %22 = fhe.sub(%a_id, %21) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %23 = fhe.multiply(%22, %22) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %24 = fhe.sub(%cst, %23) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %25 = fhe.rotate(%24) by 1 : <8 x i16>
    %26 = fhe.multiply(%24, %25) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %27 = fhe.sub(%cst, %26) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    
    // compute unique for b[0]
    %29 = fhe.rotate(%27) by 7 : <8 x i16>
    %30 = fhe.rotate(%20) by 7 : <8 x i16>
    %31 = fhe.rotate(%6) by 7 : <8 x i16>
    %32 = fhe.rotate(%13) by 7 : <8 x i16>
    // unique * b[0]
    %33 = fhe.multiply(%b_data_resized, %29, %30, %31, %32) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    // no rotate necessary

    // compute unique for b[1]
    %34 = fhe.rotate(%20) by 6 : <8 x i16>
    %35 = fhe.rotate(%13) by 6 : <8 x i16>
    %36 = fhe.rotate(%27) by 6 : <8 x i16>
    %37 = fhe.rotate(%6) by 6 : <8 x i16>
    %38 = fhe.multiply(%b_data_resized, %34, %35, %36, %37) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
      // unique * b[1]
    %51 = fhe.rotate(%38) by 7 : <8 x i16>

    // compute unique for b[2]
    %39 = fhe.rotate(%13) by 5 : <8 x i16>
    %40 = fhe.rotate(%6) by 5 : <8 x i16>
    %41 = fhe.rotate(%20) by 5 : <8 x i16>
    %42 = fhe.rotate(%27) by 5 : <8 x i16> 
    // unique * b[2]   
    %43 = fhe.multiply(%b_data_resized, %39, %40, %41, %42) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %50 = fhe.rotate(%43) by 6 : <8 x i16>

    // compute unique for b[3]
    %44 = fhe.rotate(%6) by 4 : <8 x i16>
    %45 = fhe.rotate(%27) by 4 : <8 x i16>
    %46 = fhe.rotate(%13) by 4 : <8 x i16>
    %47 = fhe.rotate(%20) by 4 : <8 x i16>
    // unique * b[3]
    %48 = fhe.multiply(%b_data_resized, %44, %45, %46, %47) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    %49 = fhe.rotate(%48) by 5 : <8 x i16>
          
   
    // running sum of loop (unique * b[i]) + sum of a[i] from before
    %57 = fhe.add(%49, %50, %51, %33, %56) : (!fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>, !fhe.batched_secret<8 x i16>) -> !fhe.batched_secret<8 x i16>
    
    %58 = fhe.extract %57[0] : <8 x i16>
    return %58 : !fhe.secret<i16>
  }
}

