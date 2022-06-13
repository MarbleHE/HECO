# Toolchain Tests
This directory contains tests for the compilation toolchain.

## Structure
There are two types of tests: some are standard unit tests written in C++ using [Google Test](https://github.com/google/googletest/blob/main/docs/primer.md) and [Google Mock](https://github.com/google/googletest/blob/main/docs/gmock_for_dummies.md),
while others use [lit (LLVM integrated tester)](https://llvm.org/docs/CommandGuide/lit.html).
lit allows specifying tests in-line in `*.mlir` files (using a `RUN ...` line).
This is used to run [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tests 
that check MLIR lowerings/conversions, but can also be used to simply verify that an `*.mlir` file parses correctly.
This setup seems to have been originally designed for regression tests in LLVM, so see [Regression Test Structure](https://llvm.org/docs/TestingGuide.html#regression-test-structure) for information on how this works.


## Running Tests

### Google Test
Please run these as usual, by compiling the appropriate target and running the resulting binary. 
Alternatively, you can use your IDE (potentially requiring a plugin) to run these.
Note that at the point of writing this, there are no C++ unit tests in this project and only lit based tests.

### lit (LLVM integrated tester) Tests
There is a "check-toolchain" target in the CMake file which can be used to run these tests.
This will search for `*.mlir` files and try to run then.
Note that the test happen during the *build* phase of the CMake process, there is no way of *running* these targets.