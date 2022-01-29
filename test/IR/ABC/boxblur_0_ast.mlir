// RUN: abc-opt -ast2ssa < %s | FileCheck %s
// std::vector<int> boxBlur(const std::vector<int> &img) {
//   const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
//   std::vector<int> img2(img.begin(), img.end());
//   for (int x = 0; x < imgSize; ++x) {
//     for (int y = 0; y < imgSize; ++y) {
//       int value = 0;
//       for (int j = -1; j < 2; ++j) {
//         for (int i = -1; i < 2; ++i) {
//           value += img.at(((x + i)*imgSize + (y + j))%img.size());
//         }
//       }
//       img2[imgSize*x + y] = value;
//     }
//   }
//   return img2;
// }
builtin.module  {
    abc.function tensor<64x!fhe.secret<f64>> @encryptedBoxBlur  {
        abc.function_parameter tensor<64x!fhe.secret<f64>> @img
    },{
        abc.block  {
            // img_size = 8
            abc.variable_declaration index @img_size = ( {
                abc.literal_int 8
            })
            // img_length = 64
            abc.variable_declaration index @img_length = ( {
                abc.literal_int 64
            })
            // img2 = img
            abc.variable_declaration tensor<64x!fhe.secret<f64>> @img2 = ( {
                abc.variable @img
            })
            // TODO: Find a way to express the loop bounds using variables? Or just go back standard abc.for?
            // for x in 0..img_size
            abc.simple_for @x = [0, 8] {
                abc.block  {
                    // for x in 0..img_size
                    abc.simple_for @y = [0, 8] {
                         abc.block  {
                            // f64 value = 0;
                            abc.variable_declaration !fhe.secret<f64> @value = ( {
                                abc.literal_int 0
                            })
                            // start looping over kernel
                            abc.simple_for @j = [-1, 2] {
                                abc.block  {
                                    abc.simple_for @i =[-1, 2] {
                                        abc.block  {
                                            // value = value + img[((x+i) * img_size + (y+j))%img_length]
                                            abc.assignment {
                                                abc.variable @value
                                            }, {
                                                // value + img[((x+i) *img_size + (y+j))%img_length]
                                                abc.binary_expression "+" {
                                                    abc.variable @value
                                                }, {
                                                    abc.index_access {
                                                        abc.variable @img
                                                    }, {
                                                        // ((x+i) *img_size + (y+j))%img_length
                                                        abc.binary_expression "%" {
                                                            abc.binary_expression "+" {
                                                                abc.binary_expression "*" {
                                                                    abc.binary_expression "+" {
                                                                        abc.variable @x
                                                                    }, {
                                                                        abc.variable @i
                                                                    }
                                                                }, {
                                                                    abc.variable @img_size
                                                                }
                                                            }, {
                                                                abc.binary_expression "+" {
                                                                    abc.variable @y
                                                                }, {
                                                                    abc.variable @j
                                                                }
                                                            }
                                                        }, {
                                                            abc.variable @img_length
                                                        }
                                                    }
                                                }
                                            }
                                        } // i loop
                                    }
                                }
                            } //j loop
                            // img2[img_size*x + y] = value
                            abc.assignment {
                                abc.index_access {
                                    abc.variable @img2
                                }, {
                                    // img_size*x + y
                                    abc.binary_expression "+" {
                                        //img_size*x
                                        abc.binary_expression "*" {
                                            abc.variable @img_size
                                        }, {
                                            abc.variable @x
                                        }
                                    }, {
                                        abc.variable @y
                                    }
                                }
                            }, {
                                abc.variable @value
                            }
                        }
                    } //y loop
                }
            } //x loop
            abc.return  {
                abc.variable @img2
            }
        }
    }
}

// CHECK: module  {
// CHECK:   func private @encryptedBoxBlur(%arg0: tensor<64x!fhe.secret<f64>>) -> tensor<64x!fhe.secret<f64>> {
// CHECK:     %c8 = arith.constant 8 : index
// CHECK:     %c64 = arith.constant 64 : index
// CHECK:     %0:4 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %c8, %arg3 = %c64, %arg4 = %arg0, %arg5 = %arg0) -> (index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>) {
// CHECK:       %1:5 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg2, %arg8 = %arg3, %arg9 = %arg1, %arg10 = %arg4, %arg11 = %arg5) -> (index, index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>) {
// CHECK:         %c0 = arith.constant 0 : index
// CHECK:         %c0_sf64 = fhe.constant 0.000000e+00 : f64
// CHECK:         %2:7 = affine.for %arg12 = -1 to 2 iter_args(%arg13 = %arg7, %arg14 = %arg8, %arg15 = %arg6, %arg16 = %arg9, %arg17 = %arg10, %arg18 = %arg11, %arg19 = %c0_sf64) -> (index, index, index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>, !fhe.secret<f64>) {
// CHECK:           %6:8 = affine.for %arg20 = -1 to 2 iter_args(%arg21 = %arg13, %arg22 = %arg14, %arg23 = %arg15, %arg24 = %arg12, %arg25 = %arg16, %arg26 = %arg17, %arg27 = %arg18, %arg28 = %arg19) -> (index, index, index, index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>, !fhe.secret<f64>) {
// CHECK:             %7 = arith.addi %arg25, %arg20 : index
// CHECK:             %8 = arith.muli %7, %arg21 : index
// CHECK:             %9 = arith.addi %arg23, %arg24 : index
// CHECK:             %10 = arith.addi %8, %9 : index
// CHECK:             %11 = arith.remui %10, %arg22 : index
// CHECK:             %12 = tensor.extract %arg27[%11] : tensor<64x!fhe.secret<f64>>
// CHECK:             %13 = fhe.add(%arg28, %12) : (!fhe.secret<f64>, !fhe.secret<f64>) -> !fhe.secret<f64>
// CHECK:             affine.yield %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %13 : index, index, index, index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>, !fhe.secret<f64>
// CHECK:           }
// CHECK:           affine.yield %6#0, %6#1, %6#2, %6#4, %6#5, %6#6, %6#7 : index, index, index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>, !fhe.secret<f64>
// CHECK:         }
// CHECK:         %3 = arith.muli %2#0, %2#3 : index
// CHECK:         %4 = arith.addi %3, %2#2 : index
// CHECK:         %5 = tensor.insert %2#6 into %2#4[%4] : tensor<64x!fhe.secret<f64>>
// CHECK:         affine.yield %2#0, %2#1, %2#3, %5, %2#5 : index, index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>
// CHECK:       }
// CHECK:       affine.yield %1#0, %1#1, %1#3, %1#4 : index, index, tensor<64x!fhe.secret<f64>>, tensor<64x!fhe.secret<f64>>
// CHECK:     }
// CHECK:     return %0#2 : tensor<64x!fhe.secret<f64>>
// CHECK:   }
// CHECK: }
