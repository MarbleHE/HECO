// Expected AST for BoxBlurTest
// TODO: Using index for all ints is a nasty hack and needs to be replaced by a proper type system!
builtin.module  {
    abc.function tensor<64xindex> @encryptedBoxBlur  {
        abc.function_parameter tensor<64xindex> @img
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

            // weights1 = {{0, 1},{0, -1}};
            abc.variable_declaration tensor<2x2xindex> @weights1  = ( {
                abc.literal_tensor dense<[[0,1],[0,-1]]> : tensor<2x2xindex>
            })


            // img2 = img
            abc.variable_declaration tensor<64xindex> @img2 = ( {
                abc.variable @img
            })
            // TODO: Find a way to express the loop bounds using variables? Or just go back standard abc.for?
            // for x in 0..img_size
            abc.simple_for @x = [0, 8] {
                abc.block  {
                    // for x in 0..img_size
                    abc.simple_for @y = [0, 8] {
                         abc.block  {
                            // int value = 0;
                            abc.variable_declaration i64 @value = ( {
                                abc.literal_int 0
                            })
                            // start looping over kernel
                            abc.simple_for @j = [-1, 1] {
                                abc.block  {
                                    abc.simple_for @i =[-1, 1] {
                                        abc.block  {
                                            //value = value + weights1.at(i + 1).at(j + 1) *img.at(((x + i)*imgSize + (y + j))%img.size())
                                            abc.assignment {
                                                abc.variable @value
                                            }, {
                                                // value + weights1.at(i + 1).at(j + 1) * img.at(((x + i)*imgSize + (y + j))%img.size())
                                                abc.binary_expression "+" {
                                                    abc.variable @value
                                                }, {
                                                    // weights1.at(i + 1).at(j + 1) * img.at(((x + i)*imgSize + (y + j))%img.size())
                                                    abc.binary_expression "*" {
                                                        // weights1.at(i + 1).at(j + 1)
                                                        abc.index_access {
                                                            abc.index_access {
                                                                abc.variable @weights1
                                                            }, {
                                                                abc.literal_int 0
                                                            }
                                                        }, {
                                                            abc.literal_int 0
                                                        }
                                                    }, {
                                                        // img.at(((x + i)*imgSize + (y + j))%img.size())
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
