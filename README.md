```
    ___       ____        __       __    _                ______                      _ __         
   /   |     / __ )____ _/ /______/ /_  (_)___  ____ _   / ____/___  ____ ___  ____  (_) /__  _____
  / /| |    / __  / __ `/ __/ ___/ __ \/ / __ \/ __ `/  / /   / __ \/ __ `__ \/ __ \/ / / _ \/ ___/
 / ___ |   / /_/ / /_/ / /_/ /__/ / / / / / / / /_/ /  / /___/ /_/ / / / / / / /_/ / / /  __/ /    
/_/  |_|  /_____/\__,_/\__/\___/_/ /_/_/_/ /_/\__, /   \____/\____/_/ /_/ /_/ .___/_/_/\___/_/     
                                             /____/                        /_/                     
```
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![CPP_Standard](https://img.shields.io/badge/c%2B%2B-11/14/17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![CI/CD](https://github.com/MarbleHE/ABC/workflows/build_run_tests/badge.svg)](https://github.com/MarbleHE/AST-Optimizer/actions)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](http://marblehe.github.io/ABC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ABC is an optimizing compiler for Fully Homomorphic Encryption (FHE). 
FHE allows computation over encrypted data, but imposes a variety of cryptographic and engineering challenges.
This compiler translates high-level program descriptions in a C-like language into the circuit-based programming paradigm of FHE.
It does so while automating as many aspects of the development as possible,
including automatically identifying and exploiting opportunities to use the powerful SIMD parallelism ("batching") present in many schemes. 

- [Repository's Structure](#repositorys-structure)
- [Compilation](#compilation)
  <!--
    - [Compile-Time Expression Simplifier](#compile-time-expression-simplifier)
    - [Cone-Rewriting](#cone-rewriting)
    - [CKKS Scheme-Specific Optimizations](#ckks-scheme-specific-optimizations)
  -->
- [Getting Started](#getting-started)
- [AST Representation](#ast-representation)
- [Extending the Library](#extending-the-library)
  - [Code Style](#code-style)
  - [Inspections](#inspections)
  - [Documentation](#documentation)
  - [Testing](#testing)
- [References](#references)

## Repository's Structure

The repository is organized as follow:

```
examples        – simple examples (WIP)
include         – header files (.h)
 └ ast          – classes to represents programs as an AST 
 └ parser       – tokenizer and parser
 └ runtime      - the runtime system (execute an AST using an FHE library)
 └ utilities    – utilities for certain tasks on ASTs (e.g., printing in DOT language)
 └ visitor      – various visitors performing optimizations
libs            – files required by CMake to download dependencies
src             – source files (.cpp)
 └ ast
 └ parser
 └ runtime
 └ utilities
 └ visitor
test             – unit tests for all classes
```

## Compilation
We include a simple C-like high-level input language, 
and a parser that translates it into Abstract Syntax Trees (ASTs), 
which form our intermediate representation (IR). 

The compilation itself can be divided into three stages:

**1. Program Transformations.** These AST-to-AST transformations aim to modify the program to make it more "FHE" friendly.
These include optimizations common in standard compilers and FHE-specific optimizations like our automated batching optimization.

**1. AST-to-Circuit Transformations.** These transform the AST into a circuit by transforming non-compatible operations (e.g., If- and While-Statements) into their circuit-equivalent using gates.
Note that instead of changing to a wires-and-gates IR, circuits are still expressed using (a subset of) the AST IR.

**3. Circuit-to-Circuit Transformations.** These transformations transform a circuit into a semantically equivalent circuit with better performance in FHE.
For example, by rewriting the circuit to reduce the multiplicative depth. This allows using smaller parameters that, in turn, enable more efficient computation.

Finally, we include a runtime system that can take (circuit-compatible) ASTs and run them against FHE libraries
(currently, only Microsoft SEAL is supported) or against a dummy scheme (for faster testing).

<!--
#### Compile-Time Expression Simplifier

The optimizer contains a compile-time expression simplifier that leverages existing knowledge (e.g., known variable values) to simplify the AST. The goal of this step is to generate an AST that is more easily transferable to a circuit. For example, the simplifier removes all variable declarations and assignments by replacing the respective variable uses with the variable's value. Also, operations that are not circuit-compatible such as If statements are rewritten. For example, an If statement of the form
``` 
IF (condition) { x = 5; } ELSE { x = 21; };
```
can be rewritten as 
```
x = (condition)*5 + (1-condition)*21;
```
without using any If statement.
Furthermore, the simplifier evaluates parts of the computation that are known at compile time, e.g., arithmetic or logical expressions. 

#### Cone-Rewriting

Our optimizer implements the heuristic approach for minimizing the multiplicative depth in Boolean circuits, as proposed by Aubry et al. [2]. This technique finds and rewrites certain structures of a circuit (so-called _cones_) in a way that the multiplicative depth is decreased locally. Repeating this procedure for all cones on the critical path leads to a global reduction of the multiplicative depth.

The approach uses four different algorithms:
An algorithm (Alg. 1) for cone construction that takes a Boolean circuit and a minimum multiplicative depth, and returns the set delta Δ of all reducible cones. This set is taken by Alg. 2 to create a circuit C^{AND} consisting of all critical AND nodes. Thereafter, a flow-based algorithm (Alg. 3) takes this C^{AND} circuit to determine Δ^{MIN}, the minimum set of cones that are required to reduce the multiplicative depth. It is desirable to minimize the number of cones because each cone rewriting adds additional nodes to the circuit. Lastly, Alg. 4 rewrites the minimum set of reducible cones while ensuring that the multiplicative depth reduced. The whole procedure is repeated until there are no reducible cones found anymore.

The implementation is heavily based on the pseudo-codes provided in the paper, except Alg. 2 (C^{AND} construction) for which no pseudo code was given. Deviant from the paper's implementation (see Sect. 3.3), we also did not consider yet the case in which no more reducible cones are available in C^{AND} but the multiplicative depth can be reduced further.

#### CKKS Scheme-Specific Optimizations

// TODO write some sentences about optimizations
-->

## Getting Started

Before starting, make sure to clone this repository using: 
```
git clone https://github.com/pjattke/msc-thesis-code-AST.git
```

The following tools are required to get this project running:
- cmake (version ≥ 3.15) to build the project
    - tested with v3.15.5
- gcc or clang to compile the sources
    - tested with Apple clang v11.0.0
- doxygen to build the documentation files (html)
    - tested with v1.8.16

The easiest way to use this library is to import the project into [CLion](https://www.jetbrains.com/clion/) which automatically loads the containing cmake build files and adds the respective targets. Development was carried out on macO (10.15.2), although the project should be running on Windows or Linux machines too.

The entire framework is built around ASTs,
the nodes of which are implemented by a class hierarchy that derives from AbstractNode (see above).
The AST nodes "own" (std::unique_ptr) their children (and, transitively, the entire subgraph).

Our input language is essentially a toy version of C.
We have a very simple hand-written parser on-top of a tokenizer (Stork) from another open source project.
It takes a string and returns the root node of the generated AST.
We also implemented the reverse, allowing us to print an AST back into this language,
which is much more human-readable than our "real" intermediate representation (JSON of the AST).

The entire compiler relies heavily on the visitor pattern,
and there is some template magic (see ScopedVisitor)
that allows one to still use overloading in visitors,
so one can e.g. handle all AbstractExpressions in one function
if the visitor only cares about statements.
This is mostly transparent to the developer.

All of our optimizations and transformations are implemented as visitors.
Since the visitor pattern doesn't easily allow you to "return" values,
we frequently use std::unordered_map<std::string, SOMETHING> ,
allowing us to associate data with (the unique ID of) an AST node.
So when visiting an If stmt, we can visit both branches recursively
and then get the "return" by looking into the hash_map at the child-nodes' ids.

Finally, our runtime system is also implemented as a visitor.
It takes an "AbstractCiphertextFactory",
the instantiations of which are basically just super thin wrappers around FHE libraries (currently just SEAL).
So if we see a multiplication node in the tree,
we just end up calling something like seal::multiply(ctxt a, ctxt b).

## AST Representation
The AST consists of nodes that are derived from either `AbstractExpression` or `AbstractStatement`,
depending on whether the operation is an expression or a statement, respectively.
```                                                                                                 
                                          ┌─────────────────────┐                                                   
                                          │    AbstractNode     │                                                   
                                          └─────────────────────┘                                                   
                                                     ▲                                                              
                                                     │                                                              
                                                     │                                                              
                         ┌─────────────────────┐     │     ┌─────────────────────┐                                  
                         │  AbstractStatement  │─────┴─────│ AbstractExpression  │                                  
                         └─────────────────────┘           └─────────────────────┘                                  
                                    ▲                                 ▲                                             
         ┌─────────────────────┐    │                                 │     ┌─────────────────────┐                 
         │     Assignment      │────┤                                 ├─────│   AbstractTarget    │                 
         └─────────────────────┘    │                                 │     └─────────────────────┘                 
                                    │                                 │                ▲                            
         ┌─────────────────────┐    │                                 │                │     ┌─────────────────────┐
         │        Block        │────┤                                 │                ├─────│  FunctionParameter  │
         └─────────────────────┘    │                                 │                │     └─────────────────────┘
                                    │                                 │                │                            
         ┌─────────────────────┐    │                                 │                │     ┌─────────────────────┐
         │         For         │────┤                                 │                ├─────│     IndexAccess     │
         └─────────────────────┘    │                                 │                │     └─────────────────────┘
                                    │                                 │                │                            
         ┌─────────────────────┐    │                                 │                │     ┌─────────────────────┐
         │      Function       │────┤                                 │                └─────│      Variable       │
         └─────────────────────┘    │                                 │                      └─────────────────────┘
                                    │                                 │                                             
         ┌─────────────────────┐    │                                 │     ┌─────────────────────┐                 
         │         If          │────┤                                 ├─────│  BinaryExpression   │                 
         └─────────────────────┘    │                                 │     └─────────────────────┘                 
                                    │                                 │                                             
         ┌─────────────────────┐    │                                 │     ┌─────────────────────┐                 
         │       Return        │────┤                                 ├─────│ OperatorExpression  │                 
         └─────────────────────┘    │                                 │     └─────────────────────┘                 
                                    │                                 │                                             
         ┌─────────────────────┐    │                                 │     ┌─────────────────────┐                 
         │ VariableDeclaration │────┘                                 ├─────│   UnaryExpression   │                 
         └─────────────────────┘                                      │     └─────────────────────┘                 
                                                                      │                                             
                                                                      │     ┌─────────────────────┐                 
                                                                      ├─────│        Call         │                 
                                                                      │     └─────────────────────┘                 
                                                                      │                                             
                                                                      │     ┌─────────────────────┐                 
                                                                      ├─────│   ExpressionList    │                 
                                                                      │     └─────────────────────┘                 
                                                                      │                                             
                                                                      │     ┌─────────────────────┐                 
                                                                      ├─────│     Literal<T>      │                 
                                                                      │     └─────────────────────┘                 
                                                                      │                                             
                                                                      │     ┌─────────────────────┐                 
                                                                      └─────│   TernaryOperator   │                 
                                                                            └─────────────────────┘                 
```
<!-- Created with monodraw, the source file is in figures/AST_class_hierarchy -->
***Figure 1:*** Class hierarchy of the AST classes.

Following, the different node types are briefly explained. The examples in brackets show how the commands would look like in "plain" C++.

- Classes derived from `AbstractExpression`
  - `BinaryExpr` – a binary arithmetic expression (e.g., `13 + 37`).
  - `Call` – a call to an internal function, i.e., its implementation is represented in the AST as a Function.
  - `FunctionParameter` – describes the parameters that a function accepts. To evaluate an AST, values must be passed for each of the parameter defined by the function's `FunctionParameter` node.
  - `Literal` – base class for all Literal derived from.
  - `LiteralBool` – models a Boolean value.
  - `LiteralInt` – models an integer value.
  - `LiteralString` – models an string value.
  - `LiteralFloat` – models a float value.
  - `LogicalExpr` – a (binary) logical expression using a logical operator (e.g., `z < 42`).
  - `UnaryExpr` – a unary expression (e.g., `!b` where `b` is a Boolean).
  - `Variable` – a variable identified by an identifier (name).
- Classes derived from `AbstractStatement`
  - `Block` – a code block `{...}`, e.g., the then-clause of an if statement.
  - `Function` – a function definition.
  - `If` – an if-conditional statement including both a then-branch and an else-branch  (e.g., `if (condition) { ... } else { ... }`), or either only a then-branch.
  - `Return` – a return statement of a method (e.g., `return` ).
  - `VarAssignm` – the assignment of a variable.
  - `VarDecl` – a variable declaration (e.g., `Z`)
  - `While` – a while-loop (e.g., `while (condition) {...}`).
  - `CallExternal` –  *see above.*
  - `Return` – a return statement of a method, i.e., the output of a computation.

<!--
As an example, the AST generated by the demo (method `generateDemoTwo`) is depicted following:

```
Function: determineSuitableX
  FunctionParameter: (encryptedA : int)
  FunctionParameter: (encryptedB : int)
  VarDecl: (randInt : int)
	BinaryExpr: 
	  CallExternal: (std::rand)
	  Operator: (mod)
	  LiteralInt: (42)
  VarDecl: (b : bool)
	LogicalExpr: 
	  Variable: (encryptedA)
	  Operator: (<)
	  LiteralInt: (2)
  VarDecl: (sum : int)
	LiteralInt: (0)
  While: 
	LogicalExpr: 
	  LogicalExpr: 
		Variable: (randInt)
		Operator: (>=)
		LiteralInt: (0)
	  Operator: (AND)
	  LogicalExpr: 
		UnaryExpr: 
		  Operator: (!)
		  Variable: (b)
		Operator: (!=)
		LiteralBool: (true)
	Block: 
	  VarAssignm: (sum)
		BinaryExpr: 
		  Variable: (sum)
		  Operator: (add)
		  Variable: (encryptedB)
	  VarAssignm: (randInt)
		BinaryExpr: 
		  Variable: (randInt)
		  Operator: (sub)
		  LiteralInt: (1)
  VarDecl: (outStr : string)
	LiteralString: (Computation finished!)
  CallExternal: (printf)
	FunctionParameter: (outStr : string)
  Return: 
	Variable: (sum)
```
-->

## Extending the Library

In general, PRs are very welcome! However, to ensure that this library keeps a high quality standard, this section introduces some conventions to be followed when extending this library.

### Code Style

The code is written in C++ and formatted according to code style file [MarbleHE_CPP_Code_Style.xml](MarbleHE_CPP_Code_Style.xml). The file can be loaded into the IDE of your choice, for example, in CLion's preferences (Editor → Code Style → C/C++). As the style file can change at any time, please keep in mind to use the latest version before sending a PR.

### Inspections

This codebase was checked against the default C/C++ inspections provided in CLion.

Further, the static code checker [cpplint](https://github.com/cpplint/cpplint) is used that provides more advanced checks. It can be integrated into CLion using the [CLion-cpplint](https://plugins.jetbrains.com/plugin/7871-clion-cpplint) plugin. To ensure consistency, pleasure use the following settings (to be provided in the plugin's options at Preferences -> cpplint option):
```
--linelength=120 --filter=-legal/copyright,-build/header_guard,-whitespace/comments,-runtime/references,-whitespace/operators
```

### Documentation

[Doxygen](http://www.doxygen.nl/manual/index.html) comments are used to create a documentation of this library.
The documentation can be generated using the supplied configuration `doxygen.conf` as described following:

```bash
doxygen Doxyfile
```

### Testing

The code is covered by unit tests to achieve high code quality and avoid introducing errors while extending the library.
For that, the [Google Test](https://github.com/google/googletest) framework is used.
The library as well as any other dependencies are automatically cloned from its GitHub repository using cmake, see [CMakeLists.txt](test/CMakeLists.txt).

The tests can be found in the [`test`](test) directory and are named according to the class file that the test covers (e.g., `MultDepthVisitorTest` for the test covering the `MultDepthVisitor` class).

It is required to submit tests for newly added features to ensure correctness and avoid breaking the feature by future changes (regression test). Any PRs without tests will not be considered to be integrated.


### Workflow
This project uses [GitHub flow](https://guides.github.com/introduction/flow/), i.e. all work around an improvement/feature happens on a specific branch and is only merged into main once it passes all checks and has been reviewed.

## References

[1] Viand, A., Shafagh, H.: [Marble: Making Fully Homomorphic Encryption Accessible to All.](http://www.vs.inf.ethz.ch/publ/papers/vianda_marble_2018.pdf) In: Proceedings of the 6th workshop on encrypted computing & applied homomorphic cryptography. pp. 49–60 (2018).

[2] Aubry, P. et al.: [Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth Minimization of Boolean Circuits.](https://eprint.iacr.org/2019/963.pdf) (2019).
