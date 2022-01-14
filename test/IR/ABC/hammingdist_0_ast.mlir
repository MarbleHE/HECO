// RUN: abc-opt -ast2ssa < %s | FileCheck %s
// int hammingDistance(const std::vector<bool> &x, const std::vector<bool> &y) {
//
//   if (x.size()!=y.size()) throw std::runtime_error("Vectors  in hamming distance must have the same length.");
//   int sum = 0;
//   for (size_t i = 0; i < x.size(); ++i) {
//     sum += x[i]!=y[i]; // or rather using NEQ = XOR = (a-b)^2
//   }
//   return sum;
// }
builtin.module  {
    abc.function !fhe.secret<f64> @encryptedHammingDistance {
        abc.function_parameter tensor<4x!fhe.secret<f64>> @x
        abc.function_parameter tensor<4x!fhe.secret<f64>> @y
    },{
        abc.block  {

            // length = 4
            abc.variable_declaration index @length = ( {
                abc.literal_int 64
            })

            // int sum = 0;
            abc.variable_declaration !fhe.secret<f64> @sum = ( {
                abc.literal_int 0
            })


            // for i in 0..length.
            // TODO: Find a way to express the loop bounds using variables? Or just go back standard abc.for?
            abc.simple_for @i = [0, 4] {
                abc.block  {

                    // sum = sum + (x[i] - y[i])*(x[i] - y[i])
                    abc.assignment {
                        abc.variable @sum
                    }, {
                        // sum + (x[i] - y[i])*(x[i] - y[i])
                        abc.binary_expression "+" {
                            abc.variable @sum
                        }, {
                            //(x[i] - y[i])*(x[i] - y[i])
                            abc.binary_expression "*" {
                                // (x[i] - y[i]))
                                abc.binary_expression "-" {
                                    // x[i]
                                    abc.index_access {
                                        abc.variable @x
                                    }, {
                                        abc.variable @i
                                    }
                                }, {
                                    // y[i]
                                    abc.index_access {
                                        abc.variable @y
                                    }, {
                                        abc.variable @i
                                    }
                                }

                            }, {
                                // (x[i] - y[i]))
                                abc.binary_expression "-" {
                                    // x[i]
                                    abc.index_access {
                                        abc.variable @x
                                    }, {
                                        abc.variable @i
                                    }
                                }, {
                                    // y[i]
                                    abc.index_access {
                                        abc.variable @y
                                    }, {
                                        abc.variable @i
                                    }
                                }
                            }
                        }
                    }
                }
            }// loop end

            //return sum
            abc.return  {
              abc.variable @sum
            }
        }
    }
}

//CHECK: module  {
//CHECK:   func private @encryptedHammingDistance(%arg0: tensor<4x!fhe.secret<f64>>, %arg1: tensor<4x!fhe.secret<f64>>) -> !fhe.secret<f64> {
//CHECK:     %c64 = arith.constant 64 : index
//CHECK:     %c0 = arith.constant 0 : index
//CHECK:     %0 = fhe.constant 0.000000e+00 : f64
//CHECK:     %1:4 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %arg1, %arg4 = %c64, %arg5 = %arg0, %arg6 = %0) -> (tensor<4x!fhe.secret<f64>>, index, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>) {
//CHECK:       %2 = tensor.extract %arg5[%arg2] : tensor<4x!fhe.secret<f64>>
//CHECK:       %3 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
//CHECK:       %4 = fhe.sub(%2, %3) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
//CHECK:       %5 = tensor.extract %arg5[%arg2] : tensor<4x!fhe.secret<f64>>
//CHECK:       %6 = tensor.extract %arg3[%arg2] : tensor<4x!fhe.secret<f64>>
//CHECK:       %7 = fhe.sub(%5, %6) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
//CHECK:       %8 = fhe.multiply(%4, %7) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
//CHECK:       %9 = fhe.add(%arg6, %8) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
//CHECK:       affine.yield %arg3, %arg4, %arg5, %9 : tensor<4x!fhe.secret<f64>>, index, tensor<4x!fhe.secret<f64>>, !fhe.secret<f64>
//CHECK:     }
//CHECK:     return %1#3 : !fhe.secret<f64>
//CHECK:   }
//CHECK: }