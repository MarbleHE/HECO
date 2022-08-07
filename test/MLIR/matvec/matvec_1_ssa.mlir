// RUN: fhe-tool -unroll-loops --canonicalize --cse < %s | FileCheck %s
// hand-written test -> prettier SSA value names
module  {
  func.func private @encryptedMVP(%m: tensor<16xf64>, %v: tensor<4x!fhe.secret<f64>>) -> tensor<4x!fhe.secret<f64>> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c0_sf64 = fhe.constant 0.000000e+00 : f64
    // for each row in matrix
    %0 = affine.for %i = 0 to 4 iter_args(%r = %v) -> (tensor<4x!fhe.secret<f64>>) {
      // iterate over the vector
      %1 = affine.for %j = 0 to 4 iter_args(%sum = %c0_sf64) -> (!fhe.secret<f64>) {
       // compute ij = i*4 + j
       %2 = arith.muli %i, %c4 : index
       %ij = arith.addi %2, %j : index
        %mij = tensor.extract %m[%ij] : tensor<16xf64>
        %vj = tensor.extract %v[%j] : tensor<4x!fhe.secret<f64>>
        %p = fhe.multiply(%mij, %vj) : (f64, !fhe.secret<f64>) -> !fhe.secret<f64>
        %s = fhe.add(%sum, %p) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
        affine.yield %s : !fhe.secret<f64>
      }
      %vnew = tensor.insert %1 into %r[%i] : tensor<4x!fhe.secret<f64>>
      affine.yield %vnew : tensor<4x!fhe.secret<f64>>
    }
    return %0: tensor<4x!fhe.secret<f64>>
  }
}

// CHECK: module  {
// CHECK:   func private @encryptedMVP(%arg0: tensor<16x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> tensor<4x!fhe.secret<f64>> {
// CHECK:     %c15 = arith.constant 15 : index
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %c14 = arith.constant 14 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c13 = arith.constant 13 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c11 = arith.constant 11 : index
// CHECK:     %c8 = arith.constant 8 : index
// CHECK:     %c10 = arith.constant 10 : index
// CHECK:     %c9 = arith.constant 9 : index
// CHECK:     %c7 = arith.constant 7 : index
// CHECK:     %c4 = arith.constant 4 : index
// CHECK:     %c6 = arith.constant 6 : index
// CHECK:     %c5 = arith.constant 5 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %0 = tensor.extract %arg0[%c0] : tensor<16x!fhe.secret<f64>>
// CHECK:     %1 = tensor.extract %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %2 = fhe.multiply(%0, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %3 = tensor.extract %arg0[%c1] : tensor<16x!fhe.secret<f64>>
// CHECK:     %4 = tensor.extract %arg1[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %5 = fhe.multiply(%3, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %6 = fhe.add(%2, %5) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %7 = tensor.extract %arg0[%c2] : tensor<16x!fhe.secret<f64>>
// CHECK:     %8 = tensor.extract %arg1[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %9 = fhe.multiply(%7, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %10 = fhe.add(%6, %9) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %11 = tensor.extract %arg0[%c3] : tensor<16x!fhe.secret<f64>>
// CHECK:     %12 = tensor.extract %arg1[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     %13 = fhe.multiply(%11, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %14 = fhe.add(%10, %13) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %15 = tensor.insert %14 into %arg1[%c0] : tensor<4x!fhe.secret<f64>>
// CHECK:     %16 = tensor.extract %arg0[%c4] : tensor<16x!fhe.secret<f64>>
// CHECK:     %17 = fhe.multiply(%16, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %18 = tensor.extract %arg0[%c5] : tensor<16x!fhe.secret<f64>>
// CHECK:     %19 = fhe.multiply(%18, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %20 = fhe.add(%17, %19) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %21 = tensor.extract %arg0[%c6] : tensor<16x!fhe.secret<f64>>
// CHECK:     %22 = fhe.multiply(%21, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %23 = fhe.add(%20, %22) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %24 = tensor.extract %arg0[%c7] : tensor<16x!fhe.secret<f64>>
// CHECK:     %25 = fhe.multiply(%24, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %26 = fhe.add(%23, %25) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %27 = tensor.insert %26 into %15[%c1] : tensor<4x!fhe.secret<f64>>
// CHECK:     %28 = tensor.extract %arg0[%c8] : tensor<16x!fhe.secret<f64>>
// CHECK:     %29 = fhe.multiply(%28, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %30 = tensor.extract %arg0[%c9] : tensor<16x!fhe.secret<f64>>
// CHECK:     %31 = fhe.multiply(%30, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %32 = fhe.add(%29, %31) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %33 = tensor.extract %arg0[%c10] : tensor<16x!fhe.secret<f64>>
// CHECK:     %34 = fhe.multiply(%33, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %35 = fhe.add(%32, %34) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %36 = tensor.extract %arg0[%c11] : tensor<16x!fhe.secret<f64>>
// CHECK:     %37 = fhe.multiply(%36, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %38 = fhe.add(%35, %37) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %39 = tensor.insert %38 into %27[%c2] : tensor<4x!fhe.secret<f64>>
// CHECK:     %40 = tensor.extract %arg0[%c12] : tensor<16x!fhe.secret<f64>>
// CHECK:     %41 = fhe.multiply(%40, %1) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %42 = tensor.extract %arg0[%c13] : tensor<16x!fhe.secret<f64>>
// CHECK:     %43 = fhe.multiply(%42, %4) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %44 = fhe.add(%41, %43) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %45 = tensor.extract %arg0[%c14] : tensor<16x!fhe.secret<f64>>
// CHECK:     %46 = fhe.multiply(%45, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %47 = fhe.add(%44, %46) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %48 = tensor.extract %arg0[%c15] : tensor<16x!fhe.secret<f64>>
// CHECK:     %49 = fhe.multiply(%48, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %50 = fhe.add(%47, %49) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:     %51 = tensor.insert %50 into %39[%c3] : tensor<4x!fhe.secret<f64>>
// CHECK:     return %51 : tensor<4x!fhe.secret<f64>>
// CHECK:   }
// CHECK: }