Attempting scalar batching for:
	rotate: %3 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %0 = tensor.extract %arg0[%c0] : tensor<16xf64>
	target_slot = 0
Attempting scalar batching for:
	rotate: %10 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %7 = tensor.extract %arg0[%c1] : tensor<16xf64>
	target_slot = 1
Attempting scalar batching for:
	rotate: %17 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %14 = tensor.extract %arg0[%c2] : tensor<16xf64>
	target_slot = 2
Attempting scalar batching for:
	rotate: %24 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %21 = tensor.extract %arg0[%c3] : tensor<16xf64>
	target_slot = 3
Attempting scalar batching for:
	rotate: %37 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %35 = tensor.extract %arg0[%c4] : tensor<16xf64>
	target_slot = 0
Attempting scalar batching for:
	rotate: %43 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %41 = tensor.extract %arg0[%c5] : tensor<16xf64>
	target_slot = 1
Attempting scalar batching for:
	rotate: %49 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %47 = tensor.extract %arg0[%c6] : tensor<16xf64>
	target_slot = 2
Attempting scalar batching for:
	rotate: %55 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %53 = tensor.extract %arg0[%c7] : tensor<16xf64>
	target_slot = 3
Attempting scalar batching for:
	rotate: %68 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %66 = tensor.extract %arg0[%c8] : tensor<16xf64>
	target_slot = 0
Attempting scalar batching for:
	rotate: %74 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %72 = tensor.extract %arg0[%c9] : tensor<16xf64>
	target_slot = 1
Attempting scalar batching for:
	rotate: %80 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %78 = tensor.extract %arg0[%c10] : tensor<16xf64>
	target_slot = 2
Attempting scalar batching for:
	rotate: %86 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %84 = tensor.extract %arg0[%c11] : tensor<16xf64>
	target_slot = 3
Attempting scalar batching for:
	rotate: %99 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %97 = tensor.extract %arg0[%c12] : tensor<16xf64>
	target_slot = 0
Attempting scalar batching for:
	rotate: %105 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %103 = tensor.extract %arg0[%c13] : tensor<16xf64>
	target_slot = 1
Attempting scalar batching for:
	rotate: %111 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %109 = tensor.extract %arg0[%c14] : tensor<16xf64>
	target_slot = 2
Attempting scalar batching for:
	rotate: %117 = fhe.rotate(%arg1) by 0 : <4 x f64>
	scalar: %115 = tensor.extract %arg0[%c15] : tensor<16xf64>
	target_slot = 3
module {
  func.func private @encryptedMVP(%arg0: tensor<16xf64>, %arg1: !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64> {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c11 = arith.constant 11 : index
    %c1 = arith.constant 1 : index
    %c13 = arith.constant 13 : index
    %c2 = arith.constant 2 : index
    %c14 = arith.constant 14 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c15 = arith.constant 15 : index
    %0 = tensor.extract %arg0[%c0] : tensor<16xf64>
    %1 = fhe.extract %arg1[0] : <4 x f64>
    %2 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %3 = fhe.multiply(%0, %2) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %4 = fhe.extract %3[0] : <4 x f64>
    %5 = linalg.init_tensor [4] : tensor<4xf64>
    %c0_0 = arith.constant 0 : index
    %6 = tensor.insert %0 into %5[%c0_0] : tensor<4xf64>
    %7 = tensor.extract %arg0[%c1] : tensor<16xf64>
    %8 = fhe.extract %arg1[1] : <4 x f64>
    %9 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %10 = fhe.multiply(%7, %9) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %11 = fhe.extract %10[1] : <4 x f64>
    %12 = linalg.init_tensor [4] : tensor<4xf64>
    %c-1 = arith.constant -1 : index
    %13 = tensor.insert %7 into %12[%c-1] : tensor<4xf64>
    %14 = tensor.extract %arg0[%c2] : tensor<16xf64>
    %15 = fhe.extract %arg1[2] : <4 x f64>
    %16 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %17 = fhe.multiply(%14, %16) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %18 = fhe.extract %17[2] : <4 x f64>
    %19 = linalg.init_tensor [4] : tensor<4xf64>
    %c-2 = arith.constant -2 : index
    %20 = tensor.insert %14 into %19[%c-2] : tensor<4xf64>
    %21 = tensor.extract %arg0[%c3] : tensor<16xf64>
    %22 = fhe.extract %arg1[3] : <4 x f64>
    %23 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %24 = fhe.multiply(%21, %23) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %25 = fhe.extract %24[3] : <4 x f64>
    %26 = linalg.init_tensor [4] : tensor<4xf64>
    %c-3 = arith.constant -3 : index
    %27 = tensor.insert %21 into %26[%c-3] : tensor<4xf64>
    %28 = fhe.rotate(%24) by -3 : <4 x f64>
    %29 = fhe.rotate(%17) by -2 : <4 x f64>
    %30 = fhe.rotate(%3) by 0 : <4 x f64>
    %31 = fhe.rotate(%10) by -1 : <4 x f64>
    %32 = fhe.add(%28, %29, %30, %31) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %33 = fhe.extract %32[0] : <4 x f64>
    %34 = fhe.insert %33 into %arg1[0] : <4 x f64>
    %35 = tensor.extract %arg0[%c4] : tensor<16xf64>
    %36 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %37 = fhe.multiply(%35, %36) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %38 = fhe.extract %37[0] : <4 x f64>
    %39 = linalg.init_tensor [4] : tensor<4xf64>
    %c0_1 = arith.constant 0 : index
    %40 = tensor.insert %35 into %39[%c0_1] : tensor<4xf64>
    %41 = tensor.extract %arg0[%c5] : tensor<16xf64>
    %42 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %43 = fhe.multiply(%41, %42) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %44 = fhe.extract %43[1] : <4 x f64>
    %45 = linalg.init_tensor [4] : tensor<4xf64>
    %c-1_2 = arith.constant -1 : index
    %46 = tensor.insert %41 into %45[%c-1_2] : tensor<4xf64>
    %47 = tensor.extract %arg0[%c6] : tensor<16xf64>
    %48 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %49 = fhe.multiply(%47, %48) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %50 = fhe.extract %49[2] : <4 x f64>
    %51 = linalg.init_tensor [4] : tensor<4xf64>
    %c-2_3 = arith.constant -2 : index
    %52 = tensor.insert %47 into %51[%c-2_3] : tensor<4xf64>
    %53 = tensor.extract %arg0[%c7] : tensor<16xf64>
    %54 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %55 = fhe.multiply(%53, %54) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %56 = fhe.extract %55[3] : <4 x f64>
    %57 = linalg.init_tensor [4] : tensor<4xf64>
    %c-3_4 = arith.constant -3 : index
    %58 = tensor.insert %53 into %57[%c-3_4] : tensor<4xf64>
    %59 = fhe.rotate(%55) by -2 : <4 x f64>
    %60 = fhe.rotate(%49) by -1 : <4 x f64>
    %61 = fhe.rotate(%37) by 1 : <4 x f64>
    %62 = fhe.rotate(%43) by 0 : <4 x f64>
    %63 = fhe.add(%59, %60, %61, %62) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %64 = fhe.extract %63[1] : <4 x f64>
    %65 = fhe.insert %64 into %34[1] : <4 x f64>
    %66 = tensor.extract %arg0[%c8] : tensor<16xf64>
    %67 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %68 = fhe.multiply(%66, %67) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %69 = fhe.extract %68[0] : <4 x f64>
    %70 = linalg.init_tensor [4] : tensor<4xf64>
    %c0_5 = arith.constant 0 : index
    %71 = tensor.insert %66 into %70[%c0_5] : tensor<4xf64>
    %72 = tensor.extract %arg0[%c9] : tensor<16xf64>
    %73 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %74 = fhe.multiply(%72, %73) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %75 = fhe.extract %74[1] : <4 x f64>
    %76 = linalg.init_tensor [4] : tensor<4xf64>
    %c-1_6 = arith.constant -1 : index
    %77 = tensor.insert %72 into %76[%c-1_6] : tensor<4xf64>
    %78 = tensor.extract %arg0[%c10] : tensor<16xf64>
    %79 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %80 = fhe.multiply(%78, %79) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %81 = fhe.extract %80[2] : <4 x f64>
    %82 = linalg.init_tensor [4] : tensor<4xf64>
    %c-2_7 = arith.constant -2 : index
    %83 = tensor.insert %78 into %82[%c-2_7] : tensor<4xf64>
    %84 = tensor.extract %arg0[%c11] : tensor<16xf64>
    %85 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %86 = fhe.multiply(%84, %85) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %87 = fhe.extract %86[3] : <4 x f64>
    %88 = linalg.init_tensor [4] : tensor<4xf64>
    %c-3_8 = arith.constant -3 : index
    %89 = tensor.insert %84 into %88[%c-3_8] : tensor<4xf64>
    %90 = fhe.rotate(%86) by -1 : <4 x f64>
    %91 = fhe.rotate(%80) by 0 : <4 x f64>
    %92 = fhe.rotate(%68) by 2 : <4 x f64>
    %93 = fhe.rotate(%74) by 1 : <4 x f64>
    %94 = fhe.add(%90, %91, %92, %93) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %95 = fhe.extract %94[2] : <4 x f64>
    %96 = fhe.insert %95 into %65[2] : <4 x f64>
    %97 = tensor.extract %arg0[%c12] : tensor<16xf64>
    %98 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %99 = fhe.multiply(%97, %98) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %100 = fhe.extract %99[0] : <4 x f64>
    %101 = linalg.init_tensor [4] : tensor<4xf64>
    %c0_9 = arith.constant 0 : index
    %102 = tensor.insert %97 into %101[%c0_9] : tensor<4xf64>
    %103 = tensor.extract %arg0[%c13] : tensor<16xf64>
    %104 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %105 = fhe.multiply(%103, %104) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %106 = fhe.extract %105[1] : <4 x f64>
    %107 = linalg.init_tensor [4] : tensor<4xf64>
    %c-1_10 = arith.constant -1 : index
    %108 = tensor.insert %103 into %107[%c-1_10] : tensor<4xf64>
    %109 = tensor.extract %arg0[%c14] : tensor<16xf64>
    %110 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %111 = fhe.multiply(%109, %110) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %112 = fhe.extract %111[2] : <4 x f64>
    %113 = linalg.init_tensor [4] : tensor<4xf64>
    %c-2_11 = arith.constant -2 : index
    %114 = tensor.insert %109 into %113[%c-2_11] : tensor<4xf64>
    %115 = tensor.extract %arg0[%c15] : tensor<16xf64>
    %116 = fhe.rotate(%arg1) by 0 : <4 x f64>
    %117 = fhe.multiply(%115, %116) : (f64, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %118 = fhe.extract %117[3] : <4 x f64>
    %119 = linalg.init_tensor [4] : tensor<4xf64>
    %c-3_12 = arith.constant -3 : index
    %120 = tensor.insert %115 into %119[%c-3_12] : tensor<4xf64>
    %121 = fhe.rotate(%117) by 0 : <4 x f64>
    %122 = fhe.rotate(%111) by 1 : <4 x f64>
    %123 = fhe.rotate(%99) by 3 : <4 x f64>
    %124 = fhe.rotate(%105) by 2 : <4 x f64>
    %125 = fhe.add(%121, %122, %123, %124) : (!fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>, !fhe.batched_secret<4 x f64>) -> !fhe.batched_secret<4 x f64>
    %126 = fhe.extract %125[3] : <4 x f64>
    %127 = fhe.insert %126 into %96[3] : <4 x f64>
    return %127 : !fhe.batched_secret<4 x f64>
  }
}

