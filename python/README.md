# Python Frontend

The current Python Frontend is an intermediate state of transitioning from executing an ABC AST to translating the Python code to an ABC dialect in MLIR.

## Limitations

Currently, the frontend does not:
- Execute any code (neither ABC AST nor the MLIR Dialect)
- Does not support all Python syntax elements (an unsupported syntax error is thrown)
- Does not handle arguments to the main function
- Does not parse functions other than `main` in `ABCContext`, and only the content of `main`.

## Getting MLIR for Python Frontend Code

CLion compiles the `pyabc` Python package and places it in `cmake-build-debug/mlir/python`. Python should find this package if it is run in this folder without the need to install `pyabc` on the local `pip` system.

The Frontend currently translates the content of the `main` function in the `ABCContext` block without translating the main function itself.

### Example

Add the following Python code to `cmake-build-debug/mlir/python/example.py`:
```Python
from pyabc import *
import logging

p = ABCProgram(logging.DEBUG)

with ABCContext(p, logging.DEBUG):
    def main():
        a = 1.0
        if a * a < a:
            return 20


        s = 0
        for i in range(10):
            s += i
        return s
```

Execute the python code:
```bash
python3 cmake-build-debug/mlir/python/test.py
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