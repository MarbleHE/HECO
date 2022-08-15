```
    __  __________________          __  ________   ______                      _ __         
   / / / / ____/ ____/ __ \   _    / / / / ____/  / ____/___  ____ ___  ____  (_) /__  _____
  / /_/ / __/ / /   / / / /  (_)  / /_/ / __/    / /   / __ \/ __ `__ \/ __ \/ / / _ \/ ___/
 / __  / /___/ /___/ /_/ /  _    / __  / /___   / /___/ /_/ / / / / / / /_/ / / /  __/ /    
/_/ /_/_____/\____/\____/  (_)  /_/ /_/_____/   \____/\____/_/ /_/ /_/ .___/_/_/\___/_/     
                                                                    /_/                     
```
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![CPP_Standard](https://img.shields.io/badge/c%2B%2B-11/14/17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![CI/CD](https://github.com/MarbleHE/ABC/workflows/build_run_tests/badge.svg)](https://github.com/MarbleHE/AST-Optimizer/actions)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](http://marblehe.github.io/HECO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HECO is an optimizing compiler for Fully Homomorphic Encryption (FHE). 
FHE allows computation over encrypted data, but imposes a variety of cryptographic and engineering challenges.
This compiler translates high-level program descriptions (expressed in Python) into the circuit-based programming paradigm of FHE.
It does so while automating as many aspects of the development as possible,
including automatically identifying and exploiting opportunities to use the powerful SIMD parallelism ("batching") present in many schemes. 

## HECO Python Frontend

> **Note**
> HECO's Python Frontend is still undergoing a major revision. 
> The current version only prints a (almost MLIR) version of the code. 
> We are working on extending the frontend with more functionality and completing the toolchain, such that frontend programs can be executed again.

### Finalize Installation

The HECO frontend relies on a development version of [xdsl](https://pypi.org/project/xdsl/). If you install HECO from PyPi, you need to install the xdsl dependency afterwards with:
```
python3 -m pip install git+https://github.com/xdslproject/xdsl.git@frontend
```

### Examples

Examples of HECO can be found in the [examples](./examples/) folder.

One of them, for computing the hamming distance of two encrypted vectors, is shown here: 
```Python
from xdsl.frontend import *
from xdsl.frontend.dialects.builtin import *
from heco.frontend.dialects.fhe import *

p = FrontendProgram()

secret_f64 = SecretType[f64]
arg_type = BatchedSecretType[f64]

with CodeContext(p):
    def encryptedHammingDistance(x: arg_type,
                                 y: arg_type) -> secret_f64:
        sum: SecretType[f64] = SecretAttr(FloatAttr(0.0))

        for idx in range(0, 4):
            sum = sum + (x[idx] - y[idx]) * (x[idx] - y[idx])

        return sum

# XXX: the part below was not yet ported to the new frontend

# Compiling FHE code
context = SEAL.BGV.new(poly_mod_degree = 1024)
f = p.compile(context = context)

# Running FHF code
x = [random.randrange(100) for _ in range(4)]
y = [random.randrange(100) for _ in range(4)]
x_enc = context.encrypt(x)
y_enc = context.encrypt(y)
s_enc = f(x_enc, y_enc)

# Verifying Result
s = context.decrypt(s_enc)
assert s == sum([(x[i] - y[i])**2 for i in range(4)])
``` 
