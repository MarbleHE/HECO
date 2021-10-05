builtin.module  {
    abc.function tensor<16384x!abc.int> @encryptedBoxBlur  {
        abc.function_parameter tensor<16384x!abc.int> @img
    },{
        abc.block  {
            abc.variable_declaration tensor<16384x!abc.int> @img2 = ( {
                abc.variable @img
            })
            affine.for %x = 0 to 128 {
                affine.for %y = 0 to 128 {
                    abc.variable_declaration tensor<16384x!abc.int> @value = ( {
                        abc.literal_int 0
                    })
                    affine.for %j = -1 to 2 {
                        affine.for %i = - 1 to 2 {
                            abc.assignment {
                                abc.variable @value
                            },  {
                                abc.binary_expression "+" {
                                    abc.variable @value
                                }, {
                                    abc.binary_expression "+" {
                                        abc.variable @value
                                    }, {
                                        abc.literal_int 5
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

