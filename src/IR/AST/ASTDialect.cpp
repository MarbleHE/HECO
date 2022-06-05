//===- ASTDialect.cpp - AST dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"

#include "heco/IR/AST/ASTDialect.h"

using namespace mlir;
using namespace ast;

bool containsExactlyOneExpressionNode(Region &region)
{

  if (region.op_begin() == region.op_end())
  {
    emitError(region.getLoc(), "Region must contain exactly one AST_ExpressionOp but is empty.");
    return false;
  }
  else if (!region.op_begin()->hasTrait<OpTrait::isAbcExpression>())
  {
    emitError(region.op_begin()->getLoc(), "Invalid op in region that should contain exactly one AST_ExpressionOp.");
    return false;
  }
  else if (++region.op_begin() != region.op_end())
  {
    emitError((++region.op_begin())->getLoc(),
              "Additional op in region that should contain exactly one AST_ExpressionOp.");
    return false;
  }
  else
  {
    return true;
  }
}

bool containsExactlyOneTargetNode(Region &region)
{

  if (region.op_begin() == region.op_end())
  {
    emitError(region.getLoc(), "Region must contain exactly one AST_TargetOp but is empty.");
    return false;
  }
  else if (!region.op_begin()->hasTrait<OpTrait::isAbcTarget>())
  {
    emitError(region.op_begin()->getLoc(), "Invalid op in region that should contain exactly one AST_TargetOp.");
    return false;
  }
  else if (++region.op_begin() != region.op_end())
  {
    emitError((++region.op_begin())->getLoc(),
              "Additional op in region that should contain exactly one AST_TargetOp.");
    return false;
  }
  else
  {
    return true;
  }
}

bool containsExactlyOneStatementNode(Region &region)
{

  if (region.op_begin() == region.op_end())
  {
    emitError(region.getLoc(), "Region must contain exactly one AST_StatementOp but is empty.");
    return false;
  }
  else if (!region.op_begin()->hasTrait<OpTrait::isAbcStatement>())
  {
    emitError(region.op_begin()->getLoc(), "Invalid op in region that should contain exactly one AST_StatementOp.");
    return false;
  }
  else if (++region.op_begin() != region.op_end())
  {
    emitError((++region.op_begin())->getLoc(),
              "Additional op in region that should contain exactly one AST_StatementOp.");
    return false;
  }
  else
  {
    return true;
  }
}

#include "heco/IR/AST/ASTDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "heco/IR/AST/ASTTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// AST dialect.
//===----------------------------------------------------------------------===//

void ASTDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "heco/IR/AST/AST.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/AST/ASTTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

//::mlir::LogicalResult RotateOp::canonicalize(RotateOp op, ::mlir::PatternRewriter &rewriter) {
//  // First check if this is a constant we can reason about statically
//  Operation *valueOp = op.rotation().getDefiningOp();
//  bool isConstLike = valueOp->hasTrait<OpTrait::ConstantLike>();
//  bool hasValue = valueOp->hasAttr("value");
//  if (isConstLike && hasValue) {
//    // Is it zero
//    bool isZero = valueOp->getAttr("value").dyn_cast<IntegerAttr>().getInt() == 0;
//    if(isZero) {
//      // No need to rotate just use the original vector
//      op.replaceAllUsesWith(op.vector());
//    }
//  }
//  return success();
//}

#define GET_OP_CLASSES
#include "heco/IR/AST/AST.cpp.inc"