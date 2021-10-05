builtin.module  {
    abc.function tensor<16384x!abc.int> @encryptedBoxBlur  {
        abc.function_parameter tensor<16xi64> @img
    },{
        abc.block  {
            abc.variable_declaration tensor<16xi64> @img2 = ( {
                abc.variable @img
            })
            abc.simple_for %x = 0 to 128 {
                abc.simple_for %y = 0 to 128 {
                    abc.variable_declaration tensor<16xi64> @value = ( {
                        abc.literal_int 0
                    })
                    abc.simple_for %j = -1 to 2 {
                        abc.simple_for %i = - 1 to 2 {
                            abc.assignment {
                                abc.variable @value
                            },  {
                                abc.binary_expression "+" {
                                    abc.variable @value
                                }, {
                                    abc.index_access "+" {
                                        abc.variable @img
                                    }, {
                                        // ((x+i) *4 + (y+j))%16


                                    }
                                }
                            }
                        }
                    }
                }
            }

            abc.return  {
                abc.variable @img2
            }
        }
  }
}

