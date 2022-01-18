//RUN: abc-opt -batching --canonicalize --cse --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64> {
    %0 = fhe.extract %arg0[55] : <f64>
    %1 = fhe.extract %arg0[63] : <f64>
    %2 = fhe.extract %arg0[7] : <f64>
    %3 = fhe.extract %arg0[56] : <f64>
    %4 = fhe.extract %arg0[0] : <f64>
    %5 = fhe.extract %arg0[8] : <f64>
    %6 = fhe.extract %arg0[57] : <f64>
    %7 = fhe.extract %arg0[1] : <f64>
    %8 = fhe.extract %arg0[9] : <f64>
    %9 = fhe.add(%8, %7, %6, %5, %4, %3, %2, %0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %10 = fhe.insert %9 into %arg0[0] : <f64>
    %11 = fhe.extract %arg0[58] : <f64>
    %12 = fhe.extract %arg0[2] : <f64>
    %13 = fhe.extract %arg0[10] : <f64>
    %14 = fhe.add(%13, %12, %11, %8, %7, %6, %5, %3, %4) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %15 = fhe.insert %14 into %10[1] : <f64>
    %16 = fhe.extract %arg0[59] : <f64>
    %17 = fhe.extract %arg0[3] : <f64>
    %18 = fhe.extract %arg0[11] : <f64>
    %19 = fhe.add(%18, %17, %16, %13, %12, %11, %8, %6, %7) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %20 = fhe.insert %19 into %15[2] : <f64>
    %21 = fhe.extract %arg0[60] : <f64>
    %22 = fhe.extract %arg0[4] : <f64>
    %23 = fhe.extract %arg0[12] : <f64>
    %24 = fhe.add(%23, %22, %21, %18, %17, %16, %13, %11, %12) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %25 = fhe.insert %24 into %20[3] : <f64>
    %26 = fhe.extract %arg0[61] : <f64>
    %27 = fhe.extract %arg0[5] : <f64>
    %28 = fhe.extract %arg0[13] : <f64>
    %29 = fhe.add(%28, %27, %26, %23, %22, %21, %18, %16, %17) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %30 = fhe.insert %29 into %25[4] : <f64>
    %31 = fhe.extract %arg0[62] : <f64>
    %32 = fhe.extract %arg0[6] : <f64>
    %33 = fhe.extract %arg0[14] : <f64>
    %34 = fhe.add(%33, %32, %31, %28, %27, %26, %23, %21, %22) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %35 = fhe.insert %34 into %30[5] : <f64>
    %36 = fhe.extract %arg0[15] : <f64>
    %37 = fhe.add(%36, %2, %1, %33, %32, %31, %28, %26, %27) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %38 = fhe.insert %37 into %35[6] : <f64>
    %39 = fhe.extract %arg0[16] : <f64>
    %40 = fhe.add(%39, %5, %4, %36, %2, %1, %33, %31, %32) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %41 = fhe.insert %40 into %38[7] : <f64>
    %42 = fhe.extract %arg0[17] : <f64>
    %43 = fhe.add(%42, %8, %7, %39, %5, %4, %36, %1, %2) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %44 = fhe.insert %43 into %41[8] : <f64>
    %45 = fhe.extract %arg0[18] : <f64>
    %46 = fhe.add(%45, %13, %12, %42, %8, %7, %39, %4, %5) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %47 = fhe.insert %46 into %44[9] : <f64>
    %48 = fhe.extract %arg0[19] : <f64>
    %49 = fhe.add(%48, %18, %17, %45, %13, %12, %42, %7, %8) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %50 = fhe.insert %49 into %47[10] : <f64>
    %51 = fhe.extract %arg0[20] : <f64>
    %52 = fhe.add(%51, %23, %22, %48, %18, %17, %45, %12, %13) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %53 = fhe.insert %52 into %50[11] : <f64>
    %54 = fhe.extract %arg0[21] : <f64>
    %55 = fhe.add(%54, %28, %27, %51, %23, %22, %48, %17, %18) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %56 = fhe.insert %55 into %53[12] : <f64>
    %57 = fhe.extract %arg0[22] : <f64>
    %58 = fhe.add(%57, %33, %32, %54, %28, %27, %51, %22, %23) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %59 = fhe.insert %58 into %56[13] : <f64>
    %60 = fhe.extract %arg0[23] : <f64>
    %61 = fhe.add(%60, %36, %2, %57, %33, %32, %54, %27, %28) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %62 = fhe.insert %61 into %59[14] : <f64>
    %63 = fhe.extract %arg0[24] : <f64>
    %64 = fhe.add(%63, %39, %5, %60, %36, %2, %57, %32, %33) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %65 = fhe.insert %64 into %62[15] : <f64>
    %66 = fhe.extract %arg0[25] : <f64>
    %67 = fhe.add(%66, %42, %8, %63, %39, %5, %60, %2, %36) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %68 = fhe.insert %67 into %65[16] : <f64>
    %69 = fhe.extract %arg0[26] : <f64>
    %70 = fhe.add(%69, %45, %13, %66, %42, %8, %63, %5, %39) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %71 = fhe.insert %70 into %68[17] : <f64>
    %72 = fhe.extract %arg0[27] : <f64>
    %73 = fhe.add(%72, %48, %18, %69, %45, %13, %66, %8, %42) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %74 = fhe.insert %73 into %71[18] : <f64>
    %75 = fhe.extract %arg0[28] : <f64>
    %76 = fhe.add(%75, %51, %23, %72, %48, %18, %69, %13, %45) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %77 = fhe.insert %76 into %74[19] : <f64>
    %78 = fhe.extract %arg0[29] : <f64>
    %79 = fhe.add(%78, %54, %28, %75, %51, %23, %72, %18, %48) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %80 = fhe.insert %79 into %77[20] : <f64>
    %81 = fhe.extract %arg0[30] : <f64>
    %82 = fhe.add(%81, %57, %33, %78, %54, %28, %75, %23, %51) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %83 = fhe.insert %82 into %80[21] : <f64>
    %84 = fhe.extract %arg0[31] : <f64>
    %85 = fhe.add(%84, %60, %36, %81, %57, %33, %78, %28, %54) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %86 = fhe.insert %85 into %83[22] : <f64>
    %87 = fhe.extract %arg0[32] : <f64>
    %88 = fhe.add(%87, %63, %39, %84, %60, %36, %81, %33, %57) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %89 = fhe.insert %88 into %86[23] : <f64>
    %90 = fhe.extract %arg0[33] : <f64>
    %91 = fhe.add(%90, %66, %42, %87, %63, %39, %84, %36, %60) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %92 = fhe.insert %91 into %89[24] : <f64>
    %93 = fhe.extract %arg0[34] : <f64>
    %94 = fhe.add(%93, %69, %45, %90, %66, %42, %87, %39, %63) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %95 = fhe.insert %94 into %92[25] : <f64>
    %96 = fhe.extract %arg0[35] : <f64>
    %97 = fhe.add(%96, %72, %48, %93, %69, %45, %90, %42, %66) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %98 = fhe.insert %97 into %95[26] : <f64>
    %99 = fhe.extract %arg0[36] : <f64>
    %100 = fhe.add(%99, %75, %51, %96, %72, %48, %93, %45, %69) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %101 = fhe.insert %100 into %98[27] : <f64>
    %102 = fhe.extract %arg0[37] : <f64>
    %103 = fhe.add(%102, %78, %54, %99, %75, %51, %96, %48, %72) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %104 = fhe.insert %103 into %101[28] : <f64>
    %105 = fhe.extract %arg0[38] : <f64>
    %106 = fhe.add(%105, %81, %57, %102, %78, %54, %99, %51, %75) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %107 = fhe.insert %106 into %104[29] : <f64>
    %108 = fhe.extract %arg0[39] : <f64>
    %109 = fhe.add(%108, %84, %60, %105, %81, %57, %102, %54, %78) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %110 = fhe.insert %109 into %107[30] : <f64>
    %111 = fhe.extract %arg0[40] : <f64>
    %112 = fhe.add(%111, %87, %63, %108, %84, %60, %105, %57, %81) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %113 = fhe.insert %112 into %110[31] : <f64>
    %114 = fhe.extract %arg0[41] : <f64>
    %115 = fhe.add(%114, %90, %66, %111, %87, %63, %108, %60, %84) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %116 = fhe.insert %115 into %113[32] : <f64>
    %117 = fhe.extract %arg0[42] : <f64>
    %118 = fhe.add(%117, %93, %69, %114, %90, %66, %111, %63, %87) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %119 = fhe.insert %118 into %116[33] : <f64>
    %120 = fhe.extract %arg0[43] : <f64>
    %121 = fhe.add(%120, %96, %72, %117, %93, %69, %114, %66, %90) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %122 = fhe.insert %121 into %119[34] : <f64>
    %123 = fhe.extract %arg0[44] : <f64>
    %124 = fhe.add(%123, %99, %75, %120, %96, %72, %117, %69, %93) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %125 = fhe.insert %124 into %122[35] : <f64>
    %126 = fhe.extract %arg0[45] : <f64>
    %127 = fhe.add(%126, %102, %78, %123, %99, %75, %120, %72, %96) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %128 = fhe.insert %127 into %125[36] : <f64>
    %129 = fhe.extract %arg0[46] : <f64>
    %130 = fhe.add(%129, %105, %81, %126, %102, %78, %123, %75, %99) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %131 = fhe.insert %130 into %128[37] : <f64>
    %132 = fhe.extract %arg0[47] : <f64>
    %133 = fhe.add(%132, %108, %84, %129, %105, %81, %126, %78, %102) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %134 = fhe.insert %133 into %131[38] : <f64>
    %135 = fhe.extract %arg0[48] : <f64>
    %136 = fhe.add(%135, %111, %87, %132, %108, %84, %129, %81, %105) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %137 = fhe.insert %136 into %134[39] : <f64>
    %138 = fhe.extract %arg0[49] : <f64>
    %139 = fhe.add(%138, %114, %90, %135, %111, %87, %132, %84, %108) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %140 = fhe.insert %139 into %137[40] : <f64>
    %141 = fhe.extract %arg0[50] : <f64>
    %142 = fhe.add(%141, %117, %93, %138, %114, %90, %135, %87, %111) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %143 = fhe.insert %142 into %140[41] : <f64>
    %144 = fhe.extract %arg0[51] : <f64>
    %145 = fhe.add(%144, %120, %96, %141, %117, %93, %138, %90, %114) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %146 = fhe.insert %145 into %143[42] : <f64>
    %147 = fhe.extract %arg0[52] : <f64>
    %148 = fhe.add(%147, %123, %99, %144, %120, %96, %141, %93, %117) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %149 = fhe.insert %148 into %146[43] : <f64>
    %150 = fhe.extract %arg0[53] : <f64>
    %151 = fhe.add(%150, %126, %102, %147, %123, %99, %144, %96, %120) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %152 = fhe.insert %151 into %149[44] : <f64>
    %153 = fhe.extract %arg0[54] : <f64>
    %154 = fhe.add(%153, %129, %105, %150, %126, %102, %147, %99, %123) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %155 = fhe.insert %154 into %152[45] : <f64>
    %156 = fhe.add(%0, %132, %108, %153, %129, %105, %150, %102, %126) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %157 = fhe.insert %156 into %155[46] : <f64>
    %158 = fhe.add(%3, %135, %111, %0, %132, %108, %153, %105, %129) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %159 = fhe.insert %158 into %157[47] : <f64>
    %160 = fhe.add(%6, %138, %114, %3, %135, %111, %0, %108, %132) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %161 = fhe.insert %160 into %159[48] : <f64>
    %162 = fhe.add(%11, %141, %117, %6, %138, %114, %3, %111, %135) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %163 = fhe.insert %162 into %161[49] : <f64>
    %164 = fhe.add(%16, %144, %120, %11, %141, %117, %6, %114, %138) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %165 = fhe.insert %164 into %163[50] : <f64>
    %166 = fhe.add(%21, %147, %123, %16, %144, %120, %11, %117, %141) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %167 = fhe.insert %166 into %165[51] : <f64>
    %168 = fhe.add(%26, %150, %126, %21, %147, %123, %16, %120, %144) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %169 = fhe.insert %168 into %167[52] : <f64>
    %170 = fhe.add(%31, %153, %129, %26, %150, %126, %21, %123, %147) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %171 = fhe.insert %170 into %169[53] : <f64>
    %172 = fhe.add(%1, %0, %132, %31, %153, %129, %26, %126, %150) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %173 = fhe.insert %172 into %171[54] : <f64>
    %174 = fhe.add(%4, %3, %135, %1, %0, %132, %31, %129, %153) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %175 = fhe.insert %174 into %173[55] : <f64>
    %176 = fhe.add(%7, %6, %138, %4, %3, %135, %1, %132, %0) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %177 = fhe.insert %176 into %175[56] : <f64>
    %178 = fhe.add(%12, %11, %141, %7, %6, %138, %4, %135, %3) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %179 = fhe.insert %178 into %177[57] : <f64>
    %180 = fhe.add(%17, %16, %144, %12, %11, %141, %7, %138, %6) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %181 = fhe.insert %180 into %179[58] : <f64>
    %182 = fhe.add(%22, %21, %147, %17, %16, %144, %12, %141, %11) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %183 = fhe.insert %182 into %181[59] : <f64>
    %184 = fhe.add(%27, %26, %150, %22, %21, %147, %17, %144, %16) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %185 = fhe.insert %184 into %183[60] : <f64>
    %186 = fhe.add(%32, %31, %153, %27, %26, %150, %22, %147, %21) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %187 = fhe.insert %186 into %185[61] : <f64>
    %188 = fhe.add(%2, %1, %0, %32, %31, %153, %27, %150, %26) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %189 = fhe.insert %188 into %187[62] : <f64>
    %190 = fhe.add(%5, %4, %3, %2, %1, %0, %32, %153, %31) : (!fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
    %191 = fhe.insert %190 into %189[63] : <f64>
    return %191 : !fhe.batched_secret<f64>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64> {
// CHECK:     %0 = fhe.rotate(%arg0) {i = -9 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %1 = fhe.rotate(%arg0) {i = -1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %2 = fhe.rotate(%arg0) {i = -57 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %3 = fhe.rotate(%arg0) {i = -8 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %4 = fhe.rotate(%arg0) {i = -56 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %5 = fhe.rotate(%arg0) {i = -7 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %6 = fhe.rotate(%arg0) {i = -55 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %7 = fhe.rotate(%arg0) {i = -63 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %9 = fhe.extract %8[0] : <f64>
// CHECK:     %10 = fhe.insert %9 into %arg0[0] : <f64>
// CHECK:     %11 = fhe.rotate(%arg0) {i = -10 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %12 = fhe.rotate(%arg0) {i = -2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %13 = fhe.rotate(%arg0) {i = -58 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %14 = fhe.add(%11, %12, %13, %0, %1, %2, %3, %4, %arg0) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %15 = fhe.extract %14[0] : <f64>
// CHECK:     %16 = fhe.insert %15 into %10[1] : <f64>
// CHECK:     %17 = fhe.rotate(%arg0) {i = 8 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %18 = fhe.rotate(%arg0) {i = -48 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %19 = fhe.rotate(%arg0) {i = 1 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %20 = fhe.rotate(%arg0) {i = 9 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %21 = fhe.rotate(%arg0) {i = -47 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %22 = fhe.rotate(%arg0) {i = 2 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %23 = fhe.rotate(%arg0) {i = -46 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %24 = fhe.rotate(%arg0) {i = 10 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %25 = fhe.add(%arg0, %17, %18, %19, %20, %21, %22, %23, %24) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %26 = fhe.extract %25[11] : <f64>
// CHECK:     %27 = fhe.insert %26 into %16[2] : <f64>
// CHECK:     %28 = fhe.extract %25[12] : <f64>
// CHECK:     %29 = fhe.insert %28 into %27[3] : <f64>
// CHECK:     %30 = fhe.extract %25[13] : <f64>
// CHECK:     %31 = fhe.insert %30 into %29[4] : <f64>
// CHECK:     %32 = fhe.extract %25[14] : <f64>
// CHECK:     %33 = fhe.insert %32 into %31[5] : <f64>
// CHECK:     %34 = fhe.extract %25[15] : <f64>
// CHECK:     %35 = fhe.insert %34 into %33[6] : <f64>
// CHECK:     %36 = fhe.rotate(%arg0) {i = -16 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %37 = fhe.rotate(%arg0) {i = -15 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %38 = fhe.rotate(%arg0) {i = -14 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %39 = fhe.rotate(%arg0) {i = -62 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %40 = fhe.rotate(%arg0) {i = -6 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %41 = fhe.add(%36, %3, %arg0, %37, %5, %7, %38, %39, %40) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %42 = fhe.extract %41[0] : <f64>
// CHECK:     %43 = fhe.insert %42 into %35[7] : <f64>
// CHECK:     %44 = fhe.rotate(%arg0) {i = -17 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %45 = fhe.add(%44, %0, %1, %36, %3, %arg0, %37, %7, %5) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %46 = fhe.extract %45[0] : <f64>
// CHECK:     %47 = fhe.insert %46 into %43[8] : <f64>
// CHECK:     %48 = fhe.rotate(%arg0) {i = -18 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %49 = fhe.add(%48, %11, %12, %44, %0, %1, %36, %arg0, %3) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %50 = fhe.extract %49[0] : <f64>
// CHECK:     %51 = fhe.insert %50 into %47[9] : <f64>
// CHECK:     %52 = fhe.rotate(%arg0) {i = 16 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %53 = fhe.rotate(%arg0) {i = 17 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %54 = fhe.rotate(%arg0) {i = 18 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %55 = fhe.add(%arg0, %17, %52, %19, %20, %53, %22, %54, %24) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %56 = fhe.extract %55[19] : <f64>
// CHECK:     %57 = fhe.insert %56 into %51[10] : <f64>
// CHECK:     %58 = fhe.extract %55[20] : <f64>
// CHECK:     %59 = fhe.insert %58 into %57[11] : <f64>
// CHECK:     %60 = fhe.extract %55[21] : <f64>
// CHECK:     %61 = fhe.insert %60 into %59[12] : <f64>
// CHECK:     %62 = fhe.extract %55[22] : <f64>
// CHECK:     %63 = fhe.insert %62 into %61[13] : <f64>
// CHECK:     %64 = fhe.extract %55[23] : <f64>
// CHECK:     %65 = fhe.insert %64 into %63[14] : <f64>
// CHECK:     %66 = fhe.extract %55[24] : <f64>
// CHECK:     %67 = fhe.insert %66 into %65[15] : <f64>
// CHECK:     %68 = fhe.extract %55[25] : <f64>
// CHECK:     %69 = fhe.insert %68 into %67[16] : <f64>
// CHECK:     %70 = fhe.extract %55[26] : <f64>
// CHECK:     %71 = fhe.insert %70 into %69[17] : <f64>
// CHECK:     %72 = fhe.extract %55[27] : <f64>
// CHECK:     %73 = fhe.insert %72 into %71[18] : <f64>
// CHECK:     %74 = fhe.extract %55[28] : <f64>
// CHECK:     %75 = fhe.insert %74 into %73[19] : <f64>
// CHECK:     %76 = fhe.extract %55[29] : <f64>
// CHECK:     %77 = fhe.insert %76 into %75[20] : <f64>
// CHECK:     %78 = fhe.extract %55[30] : <f64>
// CHECK:     %79 = fhe.insert %78 into %77[21] : <f64>
// CHECK:     %80 = fhe.extract %55[31] : <f64>
// CHECK:     %81 = fhe.insert %80 into %79[22] : <f64>
// CHECK:     %82 = fhe.extract %55[32] : <f64>
// CHECK:     %83 = fhe.insert %82 into %81[23] : <f64>
// CHECK:     %84 = fhe.extract %55[33] : <f64>
// CHECK:     %85 = fhe.insert %84 into %83[24] : <f64>
// CHECK:     %86 = fhe.extract %55[34] : <f64>
// CHECK:     %87 = fhe.insert %86 into %85[25] : <f64>
// CHECK:     %88 = fhe.extract %55[35] : <f64>
// CHECK:     %89 = fhe.insert %88 into %87[26] : <f64>
// CHECK:     %90 = fhe.extract %55[36] : <f64>
// CHECK:     %91 = fhe.insert %90 into %89[27] : <f64>
// CHECK:     %92 = fhe.extract %55[37] : <f64>
// CHECK:     %93 = fhe.insert %92 into %91[28] : <f64>
// CHECK:     %94 = fhe.extract %55[38] : <f64>
// CHECK:     %95 = fhe.insert %94 into %93[29] : <f64>
// CHECK:     %96 = fhe.extract %55[39] : <f64>
// CHECK:     %97 = fhe.insert %96 into %95[30] : <f64>
// CHECK:     %98 = fhe.extract %55[40] : <f64>
// CHECK:     %99 = fhe.insert %98 into %97[31] : <f64>
// CHECK:     %100 = fhe.extract %55[41] : <f64>
// CHECK:     %101 = fhe.insert %100 into %99[32] : <f64>
// CHECK:     %102 = fhe.extract %55[42] : <f64>
// CHECK:     %103 = fhe.insert %102 into %101[33] : <f64>
// CHECK:     %104 = fhe.extract %55[43] : <f64>
// CHECK:     %105 = fhe.insert %104 into %103[34] : <f64>
// CHECK:     %106 = fhe.extract %55[44] : <f64>
// CHECK:     %107 = fhe.insert %106 into %105[35] : <f64>
// CHECK:     %108 = fhe.extract %55[45] : <f64>
// CHECK:     %109 = fhe.insert %108 into %107[36] : <f64>
// CHECK:     %110 = fhe.extract %55[46] : <f64>
// CHECK:     %111 = fhe.insert %110 into %109[37] : <f64>
// CHECK:     %112 = fhe.extract %55[47] : <f64>
// CHECK:     %113 = fhe.insert %112 into %111[38] : <f64>
// CHECK:     %114 = fhe.extract %55[48] : <f64>
// CHECK:     %115 = fhe.insert %114 into %113[39] : <f64>
// CHECK:     %116 = fhe.extract %55[49] : <f64>
// CHECK:     %117 = fhe.insert %116 into %115[40] : <f64>
// CHECK:     %118 = fhe.extract %55[50] : <f64>
// CHECK:     %119 = fhe.insert %118 into %117[41] : <f64>
// CHECK:     %120 = fhe.extract %55[51] : <f64>
// CHECK:     %121 = fhe.insert %120 into %119[42] : <f64>
// CHECK:     %122 = fhe.extract %55[52] : <f64>
// CHECK:     %123 = fhe.insert %122 into %121[43] : <f64>
// CHECK:     %124 = fhe.extract %55[53] : <f64>
// CHECK:     %125 = fhe.insert %124 into %123[44] : <f64>
// CHECK:     %126 = fhe.extract %55[54] : <f64>
// CHECK:     %127 = fhe.insert %126 into %125[45] : <f64>
// CHECK:     %128 = fhe.extract %55[55] : <f64>
// CHECK:     %129 = fhe.insert %128 into %127[46] : <f64>
// CHECK:     %130 = fhe.extract %55[56] : <f64>
// CHECK:     %131 = fhe.insert %130 into %129[47] : <f64>
// CHECK:     %132 = fhe.extract %55[57] : <f64>
// CHECK:     %133 = fhe.insert %132 into %131[48] : <f64>
// CHECK:     %134 = fhe.extract %55[58] : <f64>
// CHECK:     %135 = fhe.insert %134 into %133[49] : <f64>
// CHECK:     %136 = fhe.extract %55[59] : <f64>
// CHECK:     %137 = fhe.insert %136 into %135[50] : <f64>
// CHECK:     %138 = fhe.extract %55[60] : <f64>
// CHECK:     %139 = fhe.insert %138 into %137[51] : <f64>
// CHECK:     %140 = fhe.extract %55[61] : <f64>
// CHECK:     %141 = fhe.insert %140 into %139[52] : <f64>
// CHECK:     %142 = fhe.extract %55[62] : <f64>
// CHECK:     %143 = fhe.insert %142 into %141[53] : <f64>
// CHECK:     %144 = fhe.extract %55[63] : <f64>
// CHECK:     %145 = fhe.insert %144 into %143[54] : <f64>
// CHECK:     %146 = fhe.rotate(%arg0) {i = -54 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %147 = fhe.add(%arg0, %4, %18, %7, %6, %21, %39, %23, %146) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %148 = fhe.extract %147[0] : <f64>
// CHECK:     %149 = fhe.insert %148 into %145[55] : <f64>
// CHECK:     %150 = fhe.rotate(%arg0) {i = -49 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %151 = fhe.add(%1, %2, %150, %arg0, %4, %18, %7, %21, %6) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %152 = fhe.extract %151[0] : <f64>
// CHECK:     %153 = fhe.insert %152 into %149[56] : <f64>
// CHECK:     %154 = fhe.rotate(%arg0) {i = -50 : si32} : (!fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %155 = fhe.add(%12, %13, %154, %1, %2, %150, %arg0, %18, %4) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %156 = fhe.extract %155[0] : <f64>
// CHECK:     %157 = fhe.insert %156 into %153[57] : <f64>
// CHECK:     %158 = fhe.add(%arg0, %4, %18, %19, %6, %21, %22, %23, %146) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %159 = fhe.extract %158[3] : <f64>
// CHECK:     %160 = fhe.insert %159 into %157[58] : <f64>
// CHECK:     %161 = fhe.extract %158[4] : <f64>
// CHECK:     %162 = fhe.insert %161 into %160[59] : <f64>
// CHECK:     %163 = fhe.extract %158[5] : <f64>
// CHECK:     %164 = fhe.insert %163 into %162[60] : <f64>
// CHECK:     %165 = fhe.extract %158[6] : <f64>
// CHECK:     %166 = fhe.insert %165 into %164[61] : <f64>
// CHECK:     %167 = fhe.extract %158[7] : <f64>
// CHECK:     %168 = fhe.insert %167 into %166[62] : <f64>
// CHECK:     %169 = fhe.add(%3, %arg0, %4, %5, %7, %6, %40, %146, %39) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
// CHECK:     %170 = fhe.extract %169[0] : <f64>
// CHECK:     %171 = fhe.insert %170 into %168[63] : <f64>
// CHECK:     return %171 : !fhe.batched_secret<f64>
// CHECK:   }
// CHECK: }