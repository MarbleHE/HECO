# Toolchain Tests
This directory contains tests for the compilation toolchain.

## Frontend Tests
The frontend tests are in `python_frontend` and are designed to be used with `pytest`.

## MLIR Tests (lit (LLVM integrated tester) Tests)
The MLIR tests use [lit (LLVM integrated tester)](https://llvm.org/docs/CommandGuide/lit.html).
lit allows specifying tests in-line in `*.mlir` files (using a `RUN ...` line).
This is used to run [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tests 
that check MLIR lowerings/conversions, but can also be used to simply verify that an `*.mlir` file parses correctly.
This setup seems to have been originally designed for regression tests in LLVM, so see [Regression Test Structure](https://llvm.org/docs/TestingGuide.html#regression-test-structure) for information on how this works.

There is a "check-heco" target in the CMake file which can be used to run these tests.
This will search for `*.mlir` files and try to run then.
Note that the test happen during the *build* phase of the CMake process, there is no way of *running* these targets.

## Evaluation (Google Test)
There are some standard unit tests in `evaluation` that are written in C++ using [Google Test](https://github.com/google/googletest/blob/main/docs/primer.md) and [Google Mock](https://github.com/google/googletest/blob/main/docs/gmock_for_dummies.md) that are used for the evaluation of HECO.
Please run these as usual, by compiling the appropriate target and running the resulting binary. 
Alternatively, you can use your IDE (potentially requiring a plugin) to run these.