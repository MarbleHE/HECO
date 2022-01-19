//RUN:  abc-opt -fhe2emitc --canonicalize < %s | FileCheck %s
module  {
  func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64> {
    %0 = fhe.rotate(%arg0) by -9 : <f64>
    %1 = fhe.rotate(%arg0) by -1 : <f64>
    %2 = fhe.rotate(%arg0) by -57 : <f64>
    %3 = fhe.rotate(%arg0) by -8 : <f64>
    %4 = fhe.rotate(%arg0) by -56 : <f64>
    %5 = fhe.rotate(%arg0) by -7 : <f64>
    %6 = fhe.rotate(%arg0) by -55 : <f64>
    %7 = fhe.rotate(%arg0) by -63 : <f64>
    %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %9 = fhe.rotate(%arg0) by 1 : <f64>
    %10 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %11 = fhe.rotate(%arg0) by 7 : <f64>
    %12 = fhe.add(%0, %1, %11, %3, %arg0, %4, %5, %6, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %13 = fhe.rotate(%arg0) by 8 : <f64>
    %14 = fhe.add(%0, %1, %11, %3, %arg0, %13, %5, %6, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %15 = fhe.rotate(%arg0) by 9 : <f64>
    %16 = fhe.add(%0, %1, %11, %3, %arg0, %13, %5, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %17 = fhe.rotate(%arg0) by 55 : <f64>
    %18 = fhe.add(%17, %1, %11, %3, %arg0, %13, %5, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %19 = fhe.rotate(%arg0) by 56 : <f64>
    %20 = fhe.add(%17, %1, %11, %19, %arg0, %13, %5, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %21 = fhe.rotate(%arg0) by 57 : <f64>
    %22 = fhe.add(%17, %1, %11, %19, %arg0, %13, %21, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %23 = fhe.rotate(%arg0) by 63 : <f64>
    %24 = fhe.add(%17, %23, %11, %19, %arg0, %13, %21, %15, %9) : (!fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>, !fhe.batched_secret<f64>) -> !fhe.batched_secret<f64>
    %25 = fhe.combine(%8[0], %10[1:6], %12[7], %14[8], %16[9:54], %18[55], %20[56], %22[57:62], %24[63], %arg0) : !fhe.batched_secret<f64>
    return %25 : !fhe.batched_secret<f64>
  }
}

// TODO: emitC for fhe.combine