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
    abc.function index @encryptedHammingDistance {
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