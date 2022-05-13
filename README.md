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
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
- [Development](#Development)


# Overview
[//]: # (TODO Documentation: Write a short introduction to HECO, what is, and how to use it.)


# Installation

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


## Building

Use the following commands to build SEAL. For more build options, see the official [SEAL repo](https://github.com/Microsoft/SEAL#getting-started).

### Building the HECO Compiler
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-fhe
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

## Checking the Installation

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


## Development

### Development Environemnt
[Visual Studio Code](https://code.visualstudio.com/) is recommended. Remember to set the `-DMLIR_DIR=...` and `-DLLVM_EXTERNAL_LIT=..` options in "Settings &rarr; Workspace &rarr; Cmake: Configure Args".
The [LLVM TableGen](https://marketplace.visualstudio.com/items?itemName=jakob-erzar.llvm-tablegen) plugin provides syntax highlighting for `*.td` files, which are used to define Dialects, Types, Operations and Rewrite Rules.
The [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir) plugin provides syntax highlighting for `*.mlir` files, which are MLIR programs in textual representation.

### Repository's Structure

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

### Debugging MLIR
[//]: # (TODO Documentation: Write up how to get useful debug info out of passes)

Useful command line options for `mlir-opt`/`heco-tool` (see also [MLIR Debugging Tips](https://mlir.llvm.org/getting_started/Debugging/) and [IR Printing](https://mlir.llvm.org/docs/PassManagement/#ir-printing)):
 * `-print-ir-before-all` - Prints the IR before each pass
 * `-debug-only=dialect-conversion` - Prints some very useful information on passes and rules being applied
 * `--verify-each=0` - Turns off the verifier, allowing one to see what the (invalid) IR looks like
 * `--allow-unregistered-dialect` - Makes parser accept unknown operations (only works if they are in generic form!)


### Working with TableGen
This project uses [Table-driven Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/)
which uses LLVM's [TableGen](https://llvm.org/docs/TableGen/index.html) language/tool.
This project specifies things in `*.td` files, which are parsed by TableGen and processed by the [mlir-tblgen](https://llvm.org/docs/CommandGuide/mlir-tblgen.html) backend.
The backend generates `C++` files (headers/sources, as appropriate),
which are then included (using standard `#include`) into the headers/sources in this project.
These generation steps are triggered automatically during build through custom CMake commands.
In order to debug issues stemming from TableGen, it is important to realize that there are four different types of failures:
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