# FHE Code Optimizer

This repository hosts an optimizer for Fully Homomorphic Encryption (FHE) working on the basis of Abstract Syntax Trees (ASTs).

For future work, this optimization component should be integrated into the [Marble](https://github.com/MarbleHE/Marble) tool to enhance its produced FHE code.

## Features

The features of this tool can roughly be divided into two parts: The computation of **statistics** that are relevant for FHE-based optimizations (e.g., multiplicative depth, noise), and performing **optimizations** on the AST.

### Statistics



### Optimizations



## Getting Started

To use the library... 


## Extending the Library

### Code Style

The code is written in C++ using [K&R style](https://en.wikipedia.org/wiki/Indentation_style#K&R_style). Classes are divided into implementation and header:

```
include/ast     – contains the header files (.h)
src/ast         – contains the implementation (.cpp)
```

For auto-formatting in the CLion IDE, please use the code style definition provided in [doc/clion/codestyle](doc/clion/codestyle).

#### Documentation

[Doxygen](http://www.doxygen.nl/manual/index.html) comments are used to create a documentation of this library. 
The documentation can be generated as follows:

```bash
cd doc/doxygen
doxygen doxygen.conf
```

### Testing

