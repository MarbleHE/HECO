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

- [Overview](#Overview)
- [Using HECO](#UsingHECO)
  - [Python Frontend](#PythonFrontend)
  - [Modes](#Modes)
    - [Interactive Mode](#InteractiveMode)
    - [Transpiler Mode](#TranspilerMode)
    - [Compiler Mode](#CompilerMode)
    - [Advanced Use Cases](#AdvancedUseCases)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
- [Development](#Development)


# Overview
HECO is an end-to-end compiler for FHE that takes high-level imperative programs and emits efficient and secure FHE implementations.
Currently, it supports Ring-LWE based schemes [B](https://eprint.iacr.org/2012/078)/[FV](https://eprint.iacr.org/2012/144), [BGV](https://eprint.iacr.org/2011/277) and [CKKS](https://eprint.iacr.org/2016/421) which offer powerful [SIMD]-like operations and can _batch_ many thousands of values into a single vector-like ciphertext.

In FHE (and other advanced cryptographic techniques such as MPC or ZKP), developers must express their applications as an (arithmetic/binary) circuit. Translating a function *f* so that the resulting circuit can be evaluated efficiently is highly non-trivial and doing so manually requires significant expert knowledge. This is where FHE compilers like HECO come in, by automating the transformation of high-level programs into lower-level representations that can be evaluated using FHE.

![program/function  f --> circuit representation --> FHE Schemes](doc/fhe_function_to_circuit.jpg)

HECO's design and novel optimizations are described in the accompanying [paper](https://arxiv.org/abs/2202.01649).
In contrast to previous compilers, HECO removes a significant burden from the developer by automating the task of translating a program to the restricted SIMD programming paradigm available in FHE. This can result in speeupds by over an order of magnitude (i.e., 10x or more) when compared to a naive baseline.

HECO is built using the [MLIR](https://mlir.llvm.org/) compiler framework and follows a traditional front-, middle- and back-end architecture. It uses two Intermediate Representations (IRs) in the middle-end, High-level IR (HIR) to express programs containing control flow and an abstraction of FHE computing (`heco::fhe`). 
This is then lowered to Scheme-specific IR (SIR), with operations corresponding to the FHE schemes' underlying operations (e.g., addition, multiplication, relineraization, etc.). Currently, HECO targets [Microsoft SEAL](https://github.com/Microsoft/SEAL) as its backend. In the future, HECO will be extended with Polynomial-level IR (PIR) and RNS IR (RIR) to directly target hardware (both CPUs and dedicated FHE accelerators).

![Architecture Overview of the Dialects](doc/heco_architecture.jpg)

# Using HECO

## Python Frontend
> **Note**
> HECO's Python Frontend is undergoing major work and is therefore not currently ready to use.

```Python
from heco import *

p = HECOProgram(logging.DEBUG)

with HECOWrapper(p):
    def main(x : list[Secret[int]], y : list[Secret[int]]):
        sum = 0
        for i in range(4):
            sum += (x[i] - y[i]) * (x[i] - y[i])
        return sum

# Compiling FHE code
context = SEAL.BGV.new(poly_mod_degree = 1024)
f = p.compile(p, context = context)

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

## Modes
HECO can be used in three distinct modes, each of which target different user needs.

### Interactive Mode
In  interactive mode, an interpreter consumes both the input data and the intermediate representation. HECO performs the usual high-level optimizations and lowers the program to 
the Scheme-specific Intermediate Representation (SIR).
This is then executed by the interpreter by calling suitable functions in SEAL.

This mode is designed to be easy-to-use and to allow rapid prototyping. While there is a performance overhead due to the interpreted nature of this mode, it should be insignificant in the context of FHE computations.

> **Note**
> Interactive mode will become available when the new Python frontend is released.

### Transpiler Mode
In transpiler mode, HECO outputs a `*.cpp` source file that can be inspected or modified before compiling & linking against SEAL. HECO performs the usual high-level optimizations and lowers the program to the Scheme-specific Intermediate Representation (SIR). This is then lowered to the MLIR `emitC` Dialect, with FHE operations translated to function calls to a SEAL wrapper. The resulting IR is then translated into an actual `*.cpp` file.

Transpiler mode is designed for advanced users that want to integrate the output into larger, existing software projects and/or modify the compiled code to better match their requirements.

<details>
  <summary>Transpiler Mode Instructions</summary>

> In order to use the transpiler mode, you need to extend the default compilation pipeline (assuming you are starting with an `*.mlir` file containing HIR, this would be `fhe-tool --from-ssa-pass [filename_in].mlir`) in two ways. 
>  1. Specify the scheme (and some core parameters) to be used by adding, e.g., `--fhe2bgv=poly_mod_degree=1024` and the corresponding lowering to emitC, e.g., `--bgv2emitc`, followed by `--cse --canonicalize` to clean up redundant operations introduced by the lowering.
>  2. Translate to an actual `*.cpp` file by passing the output through  `emitc-translate`
>
> A full example might look like this:  `fhe-tool --from-ssa-pass --fhe2bgv=poly_mod_degree=1024 --cse --canonicalize --bgv2emitc [filename_in].mlir > emitc-translate > [filename_out].cpp`.
>
> In order to compile the file, you will need to include [`wrapper.cpp.inc`](test/IR/BGV/wrapper.cpp.inc) into the file and link it against SEAL (see [`CMakeLists.txt`](test/IR/BGV/CMakeLists.txt)).  Note that the current wrapper assumes (for slightly obscure reasons) that the generated code is inside a function  `seal::Ciphertext trace()`. If this was not the case for your input, you might need to adjust the wrapper. By default, it currently serializes the result of the function into a file `trace.ctxt`.
</details>

### Compiler Mode
In compiler mode, HECO outputs an exectuable. In this mode, HECO performs the usual high-level optimizations and lowers the program to the Scheme-specific Intermediate Representation (SIR). This is then lowered to LLVM IR representing function calls to SEAL's C API, which is then compiled and linked against SEAL.

Compiler mode assumes that the input to HECO is a complete program, e.g., has a valid `main()` function. As a result, any input/output behaviour must be realized through the `LoadCiphertext`/`SaveCiphertext` operations in the scheme-specific IR.

Compiler mode is designed primarily for situations where HECO-compiled applications will be automatically deployed without developer interaction, such as in continous integration or other automated tooling.

> **Note**
> Compiler mode is not yet implemented. If you require an executable, please use Transpiler mode and subsequent manual compilation & linking for now.

# Installation
HECO uses CMake as its build system for its C++ components and follows MLIR/LLVM conventions. Please see MLIR's [Getting Started](https://mlir.llvm.org/getting_started/) for more details.

## Prerequisites

### Packages

Install `cmake`, `doxygen`, and `clang`/`gcc`.
`libboost` and `python3` dev tools are needed for pybind11.  
On Ubuntu, this can be done by running the following:

```
sudo apt-get install cmake doxygen clang libboost-all-dev python3-dev`
```


### Microsoft SEAL
HECO supports the [Microsoft SEAL](https://github.com/microsoft/SEAL) FHE library as a "backend" in its `interactive` mode.
Please follow the SEAL instrutictions on how to build and install SEAL. Currently, the most recent version of SEAL with which this project was tested is `4.0.0`. 

### Getting MLIR
There seem to be no binary distrubtions of the MLIR framework, so you'll have to compile it from source following the [MLIR getting started guide](https://mlir.llvm.org/getting_started/).
Please note that you will need to pull from [this fork](https://github.com/MarbleHE/llvm-project) instead of the original llvm repo,
as this project relies on a series of fixes/workarounds that might not yet be upstreamed.
You might want to install clang, lld, ninja and optionally [ccache](https://ccache.dev/).

<details>
  <summary>MLIR Installation/Build Commands</summary>

> The following is a reasonable start for a "Developer Friendly" installation of MLIR:
>  ```sh
>  git clone https://github.com/llvm/llvm-project.git
>
>  mkdir llvm-project/build
>
>  cd llvm-project/build
>
>  cmake -G Ninja ../llvm \
>    -DLLVM_ENABLE_PROJECTS=mlir \
>    -DLLVM_BUILD_EXAMPLES=OFF \
>    -DLLVM_TARGETS_TO_BUILD=X86 \
>    -DCMAKE_BUILD_TYPE=Debug \
>    -DLLVM_ENABLE_ASSERTIONS=ON \
>    -DCMAKE_C_COMPILER=clang \
>    -DCMAKE_CXX_COMPILER=clang++ \
>    -DLLVM_CCACHE_BUILD=ON \
>    -DLLVM_INSTALL_UTILS=ON \
>    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
>
>  cmake --build . --target check-mlir mlir-tblgen
>  ```

</details>

## Building

Use the following commands to build SEAL. For more build options, see the official [SEAL repo](https://github.com/Microsoft/SEAL#getting-started).

### Building the HECO Compiler
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` (path must be absolute). To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target fhe-tool
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

Alternatively, you can open this folder in vscode. You will want to build the heco-c target for the command line compiler.


## Installing the Python Frontend

If you want to use the python frontend, you can install our package `pyabc`.

### Over CMake
The package installation is integrated in the CMakefile. You can run `make install` in `cmake-build-debug` and it will install `pyabc`.

### Manual Installation
To install `pyabc`, first build the CMake project. Next, run the following command (in this repo's root folder):
```
python3 -m pip install --user cmake-build-debug/python
```
(assuming your build folder is `cmake-build-debug`)

For a developer installation, add the `-e` option to create a symlink to the freshly built files in `cmake-build-debug/python` instead of copying them.



<!--
1. Check that the CMake project runs through without any fatal error 

    - Troubleshooting: first, try to use "Reload Cmake Project" and/or delete the `cmake-build-debug` folder to make a fresh new build.
2. Run the "testing-all" target in CLion to execute all tests and make sure they pass on your local system. Some tests are disabled.
    - Troubleshooting: if this entry is missing, do the following to add it:
      - Open the dropdown menu with "Run/Debug Configurations"
      - Select "Edit Configurations"
      - Go to Google Tests
      - Click "add new run configuration"
      - Name it "testing-all"
      - Select "resting-all" as target
      - Save it and run (play symbol) the target
-->


# Development

## Development Environemnt
[Visual Studio Code](https://code.visualstudio.com/) is recommended. Remember to set the `-DMLIR_DIR=...` and `-DLLVM_EXTERNAL_LIT=..` options in "Settings &rarr; Workspace &rarr; Cmake: Configure Args".
The [LLVM TableGen](https://marketplace.visualstudio.com/items?itemName=jakob-erzar.llvm-tablegen) plugin provides syntax highlighting for `*.td` files, which are used to define Dialects, Types, Operations and Rewrite Rules.
The [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir) plugin provides syntax highlighting for `*.mlir` files, which are MLIR programs in textual representation.

## Repository's Structure

The repository is organized as follow:

```
.github         – Continous Integration/CI) setup files
examples        – Simple Examples for both the compiler and frontend
include         – header (.h) and TableGen (.td) files
 └ IR             – contains the different dialect definitions
 └ Passes         – contains the definitions of the different transformations
python          – the python frontend
src             – source files (.cpp)
 └ IR             – implementations of additional dialect-specific functionality
 └ Passes         – implementations of the different transformations
 └ tools          – sources for the main commandline interface
test            – unit tests for all classes
```

## Development Tips for working with MLIR-based Projects
[MLIR](https://mlir.llvm.org/) is an incredibly powerful tool and makes developing optimizing compilers significantly easier. 
However, it is not necessarily an easy-to-pick-up tool, e.g., documentation for the rapdily evolving framework is not yet sufficient in many places.
This short section cannot replace a thorough familiarization with the [MLIR Guide](https://mlir.llvm.org/getting_started/).
Instead, it provides high-level summarizes to provide context to the existing documentation. 
See [test/readme.md](test/readme.md) for information on the MLIR/LLVM testing infrastructure.


### Working with TableGen
This project uses the [Operation Definition Specification](https://mlir.llvm.org/docs/OpDefinitions/) and [Table-driven Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/)
which use LLVM's [TableGen](https://llvm.org/docs/TableGen/index.html) language/tool.
This project specifies things in `*.td` files, which are parsed by TableGen and processed by the [mlir-tblgen](https://llvm.org/docs/CommandGuide/mlir-tblgen.html) backend.
The backend generates `C++` files (headers/sources, as appropriate),
which are then included (using standard `#include`) into the headers/sources in this project.
These generation steps are triggered automatically during build through custom CMake commands.

In order to debug issues stemming from TableGen, it is important to realize that there are **four different types of TableGen failures**:
* The TableGen parser throws an error like `error: expected ')' in dag init`. 
  This implies a syntax error in the `*.td` file, e.g., missing comma or parenthesis.
  These errors are presented the same as the next kind, but can be recognized since they usually mention "dag", "init" and/or "node". 
* The MLIR TableGen backend throws an error like `error: cannot bind symbol twice`. These are semantic/logical errors in your `*.td` file, or hint at missing features in the backend.
  Instead of stopping on an error, the backend might also crash with a stack dump. Scroll right to see if this is due to an assert being triggered.
  This usually indicates a bug in the backend, rather than in your code (at the very least, that an `assert` should be replaced by a `llvm::PrintFatalError`).
* The C++ compilation fails with an error in the generated `*.h.inc` or `*.cpp.inc` file. 
  This can be caused by either user error, e.g., when trying to do a rewrite that doesn't respect return types,
  or it can also be a sign of a bug in the MLIR TableGen backend.
* The project builds, but crashes during runtime. Again, this can be an indication of a backend bug or user error.

### Debugging MLIR
[//]: # (TODO Documentation: Write up how to get useful debug info out of passes)

Useful command line options for `mlir-opt`/`heco-tool` (see also [MLIR Debugging Tips](https://mlir.llvm.org/getting_started/Debugging/) and [IR Printing](https://mlir.llvm.org/docs/PassManagement/#ir-printing)):
 * `-print-ir-before-all` - Prints the IR before each pass
 * `-debug-only=dialect-conversion` - Prints some very useful information on passes and rules being applied
 * `--verify-each=0` - Turns off the verifier, allowing one to see what the (invalid) IR looks like
 * `--allow-unregistered-dialect` - Makes parser accept unknown operations (only works if they are in generic form!)