// Expected AST for BoxBlurTest
builtin.module  {
    abc.function tensor<16384x!abc.int> @encryptedBoxBlur  {
        abc.function_parameter tensor<16xi64> @img
    },{
        abc.block  {
            abc.variable_declaration tensor<16xi64> @img2 = ( {
                abc.variable @img
            })
            abc.simple_for @x = [0, 128] {
                abc.simple_for @y = [0, 128] {
                    // int value = 0;
                    abc.variable_declaration i64 @value = ( {
                        abc.literal_int 0
                    })
                    // start looping over kernel
                    abc.simple_for @j = [-1, 2] {
                        abc.simple_for @i =[-1, 2] {
                            // value = value + img[((x+i) *4 + (y+j))%16]
                            abc.assignment {
                                abc.variable @value
                            }, {
                                // value + img[((x+i) *4 + (y+j))%16]
                                abc.binary_expression "+" {
                                    abc.variable @value
                                }, {
                                    abc.index_access {
                                        abc.variable @img
                                    }, {
                                        // (((x+i) *4 + (y+j))%16
                                        abc.binary_expression "%" {
                                            abc.binary_expression "+" {
                                                abc.binary_expression "*" {
                                                    abc.binary_expression "+" {
                                                        abc.variable @x
                                                    }, {
                                                        abc.variable @i
                                                    }
                                                }, {
                                                    abc.literal_int 4
                                                }
                                            }, {
                                                abc.binary_expression "+" {
                                                    abc.variable @y
                                                }, {
                                                    abc.variable @j
                                                }
                                            }
                                        }, {
                                            abc.literal_int 16
                                        }
                                    }
                                }
                            }
                        } // i loop
                    } //j loop
                    // img2[4*x + y] = value
                    abc.assignment {
                        abc.index_access {
                            abc.variable @img2
                        }, {
                            // 4*x + y
                            abc.binary_expression "+" {
                                //4*x
                                abc.binary_expression "*" {
                                    abc.literal_int 4
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
                } //y loop
            } //x loop
            abc.return  {
                    abc.variable @img2
            }
        }
    }
}

