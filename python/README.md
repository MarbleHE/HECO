# Python Frontend

The current Python Frontend is an intermediate state of transitioning from executing an ABC AST to translating the Python code to an ABC dialect in MLIR.

Currently, we only support Python `>= 3.9`

## Installation

To install `pyabc` locally for testing purposes, run the following command to link the `pyabc` package in your local Python installation to the build directory of this project. The package is automatically the latest version (also when you recompile this project).
```bash
python3 -m pip install -e cmake-build-debug/python
```

## Limitations

Currently, the frontend does not:
- Execute any code (neither ABC AST nor the MLIR Dialect)
- Does not support all Python syntax elements (an unsupported syntax error is thrown)
- Argument type (annotations) are not considered yet. All parameters still have type "void"

## Getting MLIR for Python Frontend Code

CLion compiles the `pyabc` Python package and places it in `cmake-build-debug/python`. Python should find this package if it is run in this folder without the need to install `pyabc` on the local `pip` system.

The Frontend currently translates the content of the `main` function in the `ABCContext` block without translating the main function itself.

### Example

Execute the python code:
```bash
python3 python/examples/example_basic.py
```

Expected output:
```text
DEBUG:root:Start parsing With block at line 6
module  {
  abc.if  {
    abc.binary_expression "<"  {
      abc.binary_expression "*"  {
        abc.variable @a
      },  {
        abc.variable @a
      }
    },  {
      abc.variable @a
    }
  },  {
    abc.block  {
      abc.return  {
        abc.literal_int 20 : i64
      }
    }
  }
  abc.variable_declaration i64 @s = ( {
    abc.literal_int 0 : i64
  })
  abc.for  {
    abc.block  {
      abc.variable_declaration i64 @i = ( {
        abc.literal_int 0 : i64
      })
    }
  },  {
    abc.binary_expression "||"  {
      abc.binary_expression "&&"  {
        abc.binary_expression "<="  {
          abc.literal_int 0 : i64
        },  {
          abc.literal_int 10 : i64
        }
      },  {
        abc.binary_expression "<"  {
          abc.variable @i
        },  {
          abc.literal_int 10 : i64
        }
      }
    },  {
      abc.binary_expression "&&"  {
        abc.binary_expression ">"  {
          abc.literal_int 0 : i64
        },  {
          abc.literal_int 10 : i64
        }
      },  {
        abc.binary_expression ">"  {
          abc.variable @i
        },  {
          abc.literal_int 10 : i64
        }
      }
    }
  },  {
    abc.block  {
      abc.assignment  {
        abc.variable @i
      },  {
        abc.binary_expression "+"  {
          abc.variable @i
        },  {
          abc.literal_int 1 : i64
        }
      }
    }
  },  {
    abc.block  {
      abc.assignment  {
        abc.variable @s
      },  {
        abc.binary_expression "+"  {
          abc.variable @s
        },  {
          abc.variable @i
        }
      }
    }
  }
  abc.return  {
    abc.variable @s
  }
}
```

Remark: the for loop translation is this complicated, because it has to translate the Python AST object for `range(10)`, which contains lower and upper limits, as well as arbitrary step sizes (positive as well as negative). There is no standard way to specify a "C++-style" for loop in Python.

### Example (function call)

Execute the python code:
```bash
python3 python/examples/example_fn_call.py
```

Expected output:
```
module  {
^bb1:  // no predecessors
  abc.function none @add  {
    abc.function_parameter none @i
  ^bb1:  // no predecessors
    abc.function_parameter none @j
  },  {
    abc.block  {
      abc.return  {
        abc.binary_expression "+"  {
          abc.variable @i
        },  {
          abc.variable @j
        }
      }
    }
  }
^bb2:  // no predecessors
  abc.function none @main  {
    abc.function_parameter none @a
  },  {
    abc.block  {
      abc.return  {
        abc.call  {
          abc.variable @a
        ^bb1:  // no predecessors
          abc.literal_int 2 : i64
        } attributes {name = "add"}
      }
    }
  }
}
```
