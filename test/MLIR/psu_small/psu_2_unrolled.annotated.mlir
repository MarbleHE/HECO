module {
  func.func private @encryptedPSU(%arg0: tensor<4x2x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>, %arg2: tensor<4x2x!fhe.secret<f64>>, %arg3: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1_sf64 = fhe.constant 1.000000e+00 : f64
    
    // add all a[i]
      %0 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
      %1 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
      %2 = fhe.add(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %3 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
      %4 = fhe.add(%2, %3) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %5 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
      %6 = fhe.add(%4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    

    // a[0]
      // compute a[0] == b[0]
          %7 = tensor.extract %arg0[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
          %8 = tensor.extract %arg2[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
          %9 = fhe.sub(%7, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %10 = fhe.multiply(%9, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %11 = fhe.sub(%c1_sf64, %10) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
          %12 = tensor.extract %arg0[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
          %13 = tensor.extract %arg2[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
          %14 = fhe.sub(%12, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %15 = fhe.multiply(%14, %14) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %16 = fhe.sub(%c1_sf64, %15) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
  
        %17 = fhe.multiply(%16, %11) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %18 = fhe.sub(%c1_sf64, %17) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
      // compute a[0] == b[1]
          %19 = tensor.extract %arg0[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
          %20 = tensor.extract %arg2[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
          %21 = fhe.sub(%19, %20) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %22 = fhe.multiply(%21, %21) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %23 = fhe.sub(%c1_sf64, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
          %24 = tensor.extract %arg0[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
          %25 = tensor.extract %arg2[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
          %26 = fhe.sub(%24, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %27 = fhe.multiply(%26, %26) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %28 = fhe.sub(%c1_sf64, %27) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          
        %29 = fhe.multiply(%28, %23) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %30 = fhe.sub(%c1_sf64, %29) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
      %31 = fhe.multiply(%30, %18) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
      // compute a[0] == b[2]
          %32 = tensor.extract %arg0[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
          %33 = tensor.extract %arg2[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
          %34 = fhe.sub(%32, %33) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %35 = fhe.multiply(%34, %34) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %36 = fhe.sub(%c1_sf64, %35) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
          %37 = tensor.extract %arg0[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
          %38 = tensor.extract %arg2[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
          %39 = fhe.sub(%37, %38) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %40 = fhe.multiply(%39, %39) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %41 = fhe.sub(%c1_sf64, %40) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        
        %42 = fhe.multiply(%41, %36) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        %43 = fhe.sub(%c1_sf64, %42) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
      %44 = fhe.multiply(%43, %31) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
      // compute a[0] == b[3]
          %45 = tensor.extract %arg0[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
          %46 = tensor.extract %arg2[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
          %47 = fhe.sub(%45, %46) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %48 = fhe.multiply(%47, %47) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %49 = fhe.sub(%c1_sf64, %48) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
          %50 = tensor.extract %arg0[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
          %51 = tensor.extract %arg2[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
          %52 = fhe.sub(%50, %51) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %53 = fhe.multiply(%52, %52) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
          %54 = fhe.sub(%c1_sf64, %53) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
      %55 = fhe.multiply(%54, %49) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %56 = fhe.sub(%c1_sf64, %55) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    %57 = fhe.multiply(%56, %44) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %58 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
    %59 = fhe.multiply(%58, %57) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %60 = fhe.add(%6, %59) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
  
    // a[1]
    
      %61 = tensor.extract %arg0[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %62 = tensor.extract %arg2[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
      %63 = fhe.sub(%61, %62) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %64 = fhe.multiply(%63, %63) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %65 = fhe.sub(%c1_sf64, %64) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %66 = tensor.extract %arg0[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %67 = tensor.extract %arg2[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
      %68 = fhe.sub(%66, %67) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %69 = fhe.multiply(%68, %68) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %70 = fhe.sub(%c1_sf64, %69) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %71 = fhe.multiply(%70, %65) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %72 = fhe.sub(%c1_sf64, %71) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %73 = tensor.extract %arg0[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %74 = tensor.extract %arg2[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %75 = fhe.sub(%73, %74) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %76 = fhe.multiply(%75, %75) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %77 = fhe.sub(%c1_sf64, %76) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %78 = tensor.extract %arg0[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %79 = tensor.extract %arg2[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %80 = fhe.sub(%78, %79) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %81 = fhe.multiply(%80, %80) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %82 = fhe.sub(%c1_sf64, %81) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %83 = fhe.multiply(%82, %77) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %84 = fhe.sub(%c1_sf64, %83) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %85 = fhe.multiply(%84, %72) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %86 = tensor.extract %arg0[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %87 = tensor.extract %arg2[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %88 = fhe.sub(%86, %87) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %89 = fhe.multiply(%88, %88) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %90 = fhe.sub(%c1_sf64, %89) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %91 = tensor.extract %arg0[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %92 = tensor.extract %arg2[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %93 = fhe.sub(%91, %92) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %94 = fhe.multiply(%93, %93) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %95 = fhe.sub(%c1_sf64, %94) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %96 = fhe.multiply(%95, %90) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %97 = fhe.sub(%c1_sf64, %96) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %98 = fhe.multiply(%97, %85) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %99 = tensor.extract %arg0[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %100 = tensor.extract %arg2[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %101 = fhe.sub(%99, %100) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %102 = fhe.multiply(%101, %101) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %103 = fhe.sub(%c1_sf64, %102) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %104 = tensor.extract %arg0[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %105 = tensor.extract %arg2[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %106 = fhe.sub(%104, %105) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %107 = fhe.multiply(%106, %106) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %108 = fhe.sub(%c1_sf64, %107) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %109 = fhe.multiply(%108, %103) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %110 = fhe.sub(%c1_sf64, %109) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %111 = fhe.multiply(%110, %98) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
   
    %112 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
    %113 = fhe.multiply(%112, %111) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %114 = fhe.add(%60, %113) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    //a[2]
      %115 = tensor.extract %arg0[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %116 = tensor.extract %arg2[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
      %117 = fhe.sub(%115, %116) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %118 = fhe.multiply(%117, %117) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %119 = fhe.sub(%c1_sf64, %118) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %120 = tensor.extract %arg0[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %121 = tensor.extract %arg2[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
      %122 = fhe.sub(%120, %121) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %123 = fhe.multiply(%122, %122) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %124 = fhe.sub(%c1_sf64, %123) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %125 = fhe.multiply(%124, %119) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %126 = fhe.sub(%c1_sf64, %125) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %127 = tensor.extract %arg0[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %128 = tensor.extract %arg2[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %129 = fhe.sub(%127, %128) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %130 = fhe.multiply(%129, %129) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %131 = fhe.sub(%c1_sf64, %130) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %132 = tensor.extract %arg0[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %133 = tensor.extract %arg2[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %134 = fhe.sub(%132, %133) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %135 = fhe.multiply(%134, %134) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %136 = fhe.sub(%c1_sf64, %135) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %137 = fhe.multiply(%136, %131) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %138 = fhe.sub(%c1_sf64, %137) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %139 = fhe.multiply(%138, %126) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %140 = tensor.extract %arg0[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %141 = tensor.extract %arg2[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %142 = fhe.sub(%140, %141) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %143 = fhe.multiply(%142, %142) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %144 = fhe.sub(%c1_sf64, %143) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %145 = tensor.extract %arg0[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %146 = tensor.extract %arg2[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %147 = fhe.sub(%145, %146) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %148 = fhe.multiply(%147, %147) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %149 = fhe.sub(%c1_sf64, %148) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %150 = fhe.multiply(%149, %144) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %151 = fhe.sub(%c1_sf64, %150) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %152 = fhe.multiply(%151, %139) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %153 = tensor.extract %arg0[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %154 = tensor.extract %arg2[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %155 = fhe.sub(%153, %154) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %156 = fhe.multiply(%155, %155) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %157 = fhe.sub(%c1_sf64, %156) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %158 = tensor.extract %arg0[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %159 = tensor.extract %arg2[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %160 = fhe.sub(%158, %159) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %161 = fhe.multiply(%160, %160) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %162 = fhe.sub(%c1_sf64, %161) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %163 = fhe.multiply(%162, %157) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %164 = fhe.sub(%c1_sf64, %163) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %165 = fhe.multiply(%164, %152) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %166 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
    
    %167 = fhe.multiply(%166, %165) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %168 = fhe.add(%114, %167) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    //a[3]
      %169 = tensor.extract %arg0[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %170 = tensor.extract %arg2[%c0, %c0] : tensor<4x2x!fhe.secret<f64>>
      %171 = fhe.sub(%169, %170) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %172 = fhe.multiply(%171, %171) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %173 = fhe.sub(%c1_sf64, %172) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %174 = tensor.extract %arg0[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %175 = tensor.extract %arg2[%c0, %c1] : tensor<4x2x!fhe.secret<f64>>
      %176 = fhe.sub(%174, %175) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %177 = fhe.multiply(%176, %176) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %178 = fhe.sub(%c1_sf64, %177) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %179 = fhe.multiply(%178, %173) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %180 = fhe.sub(%c1_sf64, %179) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %181 = tensor.extract %arg0[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %182 = tensor.extract %arg2[%c1, %c0] : tensor<4x2x!fhe.secret<f64>>
      %183 = fhe.sub(%181, %182) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %184 = fhe.multiply(%183, %183) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %185 = fhe.sub(%c1_sf64, %184) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %186 = tensor.extract %arg0[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %187 = tensor.extract %arg2[%c1, %c1] : tensor<4x2x!fhe.secret<f64>>
      %188 = fhe.sub(%186, %187) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %189 = fhe.multiply(%188, %188) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %190 = fhe.sub(%c1_sf64, %189) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %191 = fhe.multiply(%190, %185) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %192 = fhe.sub(%c1_sf64, %191) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %193 = fhe.multiply(%192, %180) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %194 = tensor.extract %arg0[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %195 = tensor.extract %arg2[%c2, %c0] : tensor<4x2x!fhe.secret<f64>>
      %196 = fhe.sub(%194, %195) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %197 = fhe.multiply(%196, %196) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %198 = fhe.sub(%c1_sf64, %197) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %199 = tensor.extract %arg0[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %200 = tensor.extract %arg2[%c2, %c1] : tensor<4x2x!fhe.secret<f64>>
      %201 = fhe.sub(%199, %200) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %202 = fhe.multiply(%201, %201) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %203 = fhe.sub(%c1_sf64, %202) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %204 = fhe.multiply(%203, %198) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %205 = fhe.sub(%c1_sf64, %204) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %206 = fhe.multiply(%205, %193) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %207 = tensor.extract %arg0[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %208 = tensor.extract %arg2[%c3, %c0] : tensor<4x2x!fhe.secret<f64>>
      %209 = fhe.sub(%207, %208) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %210 = fhe.multiply(%209, %209) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %211 = fhe.sub(%c1_sf64, %210) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %212 = tensor.extract %arg0[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %213 = tensor.extract %arg2[%c3, %c1] : tensor<4x2x!fhe.secret<f64>>
      %214 = fhe.sub(%212, %213) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %215 = fhe.multiply(%214, %214) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %216 = fhe.sub(%c1_sf64, %215) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %217 = fhe.multiply(%216, %211) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %218 = fhe.sub(%c1_sf64, %217) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
      %219 = fhe.multiply(%218, %206) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    %220 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
    %221 = fhe.multiply(%220, %219) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %222 = fhe.add(%168, %221) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    
    return %222 : !fhe.secret<f64>
  }
}

