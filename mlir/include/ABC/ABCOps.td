//===- ABCOps.td - ABC dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "ABCDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Real ABC operation definitions.
//===----------------------------------------------------------------------===//

def ABC_FunctionOp : ABC_Op<"function", []> {
  let summary = "AST Node Function";
  let regions = (region AnyExpression: $parameters, AnyRegion: $body);
}


def ABC_FooOp : ABC_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `abc.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = abc.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def ConstantOp : ABC_Op<"constant", [NoSideEffect]> {
let summary = "constant";
let description = [{
Constant operation turns a literal into an SSA value. The data is attached
to the operation as an attribute. For example:

```mlir
%0 = abc.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
: tensor<2x3xf64>
```
}];

// The constant operation takes an attribute as the only input.
let arguments = (ins F64ElementsAttr:$value);

// The constant operation returns a single value of TensorType.
let results = (outs F64Tensor);

// Specify a parser and printer method.
let parser = [{ return ::parseConstantOp(parser, result); }];
let printer = [{ return ::print(p, *this); }];

// Add custom build methods for the constant operation. These method populates
// the `state` that MLIR uses to create operations, i.e. these are used when
// using `builder.create<ConstantOp>(...)`.
let builders = [
// Build a constant with a given constant tensor value.
OpBuilder<(ins "DenseElementsAttr":$value), [{
build($_builder, $_state, value.getType(), value);
}]>,

// Build a constant with a given constant floating-point value.
OpBuilder<(ins "double":$value)>
];

// Invoke a static verify method to verify this constant operation.
let verifier = [{ return ::verify(*this); }];
}

def AddOp : ABC_Op<"add"> {
let summary = "element-wise addition operation";
let description = [{
The "add" operation performs element-wise addition between two tensors.
The shapes of the tensor operands are expected to match.
}];

let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
let results = (outs F64Tensor);

// Specify a parser and printer method.
let parser = [{ return ::parseBinaryOp(parser, result); }];
let printer = [{ return ::printBinaryOp(p, *this); }];

// Allow building an AddOp with from the two input operands.
let builders = [
OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
];
}

def GenericCallOp : ABC_Op<"generic_call"> {
let summary = "generic call operation";
let description = [{
Generic calls represent calls to a user defined function that needs to
be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
arguments expected by the callee. For example:

```mlir
%4 = abc.generic_call @my_func(%1, %3)
: (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

This is only valid if a function named "my_func" exists and takes two
    arguments.
}];

// The generic call operation takes a symbol reference attribute as the
// callee, and inputs for the call.
let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

// The generic call operation returns a single value of TensorType.
let results = (outs F64Tensor);

// Specialize assembly printing and parsing using a declarative format.
let assemblyFormat = [{
$callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
}];

// Add custom build methods for the generic call operation.
let builders = [
OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
];
}

def MulOp : ABC_Op<"mul"> {
let summary = "element-wise multiplication operation";
let description = [{
The "mul" operation performs element-wise multiplication between two
tensors. The shapes of the tensor operands are expected to match.
}];

let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
let results = (outs F64Tensor);

// Specify a parser and printer method.
let parser = [{ return ::parseBinaryOp(parser, result); }];
let printer = [{ return ::printBinaryOp(p, *this); }];

// Allow building a MulOp with from the two input operands.
let builders = [
OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
];
}

def PrintOp : ABC_Op<"print"> {
let summary = "print operation";
let description = [{
The "print" builtin operation prints a given input tensor, and produces
    no results.
}];

// The print operation takes an input tensor to print.
let arguments = (ins F64Tensor:$input);

let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReturnOp : ABC_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
    Terminator]> {
let summary = "return operation";
let description = [{
The "return" operation represents a return operation within a function.
The operation takes an optional tensor operand and produces no results.
The operand type must match the signature of the function that contains
the operation. For example:

```mlir
    func @foo() -> tensor<2xf64> {
...
abc.return %0 : tensor<2xf64>
}
```
}];

// The return operation takes an optional input operand to return. This
// value must match the return type of the enclosing function.
let arguments = (ins Variadic<F64Tensor>:$input);

// The return operation only emits the input in the format if it is present.
let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

// Allow building a ReturnOp with no return operand.
let builders = [
OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
];

// Provide extra utility definitions on the c++ operation class definition.
let extraClassDeclaration = [{
bool hasOperand() { return getNumOperands() != 0; }
}];

// Invoke a static verify method to verify this return operation.
let verifier = [{ return ::verify(*this); }];
}


#endif // STANDALONE_OPS
