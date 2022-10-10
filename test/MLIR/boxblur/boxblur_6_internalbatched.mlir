//RUN: fhe-tool --fhe2bgv=poly_mod_degree=1024 --cse --canonicalize  < %s | FileCheck %s
module {
  func.func private @encryptedBoxBlur(%arg0: !fhe.batched_secret<64 x i16>) -> !fhe.batched_secret<64 x i16> {
    %0 = fhe.rotate(%arg0) by 55 : <64 x i16>
    %1 = fhe.rotate(%arg0) by 63 : <64 x i16>
    %2 = fhe.rotate(%arg0) by 7 : <64 x i16>
    %3 = fhe.rotate(%arg0) by 56 : <64 x i16>
    %4 = fhe.rotate(%arg0) by 8 : <64 x i16>
    %5 = fhe.rotate(%arg0) by 57 : <64 x i16>
    %6 = fhe.rotate(%arg0) by 9 : <64 x i16>
    %7 = fhe.rotate(%arg0) by 1 : <64 x i16>
    %8 = fhe.add(%0, %1, %2, %3, %arg0, %4, %5, %6, %7) : (!fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>, !fhe.batched_secret<64 x i16>) -> !fhe.batched_secret<64 x i16>
    return %8 : !fhe.batched_secret<64 x i16>
  }
}

