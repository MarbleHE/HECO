//===- ABCDialect.h - ABC dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STANDALONEDIALECT_H
#define STANDALONE_STANDALONEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace OpTrait {
template<typename ConcreteType>
class isAbcExpression : public mlir::OpTrait::TraitBase<ConcreteType, isAbcExpression> {
};

template<typename ConcreteType>
class isAbcStatement : public mlir::OpTrait::TraitBase<ConcreteType, isAbcStatement> {
};

template<typename ConcreteType>
class isAbcTarget : public mlir::OpTrait::TraitBase<ConcreteType, isAbcTarget> {
};
}
}


#include "ABC/ABCOpsDialect.h.inc"

bool containsExactlyOneExpressionNode(mlir::Region& region);

bool containsExactlyOneTargetNode(mlir::Region& region);

bool containsExactlyOneStatementNode(mlir::Region& region);


#define GET_OP_CLASSES
#include "ABC/ABCOps.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ABC/ABCOpsTypes.h.inc"

#endif // STANDALONE_STANDALONEDIALECT_H
