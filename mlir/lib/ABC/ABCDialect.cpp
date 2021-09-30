//===- ABCDialect.cpp - ABC dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABC/ABCDialect.h"

using namespace mlir;
using namespace mlir::abc;

bool containsExactlyOneExpressionNode(Region &region) {

  if (region.op_begin()==region.op_end()) {
    emitError(region.getLoc(), "Region must contain exactly one ABC_ExpressionOp but is empty.");
    return false;
  } else if (!region.op_begin()->hasTrait<OpTrait::isAbcExpression>()) {
    emitError(region.op_begin()->getLoc(), "Invalid op in region that should contain exactly one ABC_ExpressionOp.");
    return false;
  } else if (++region.op_begin()!=region.op_end()) {
    emitError((++region.op_begin())->getLoc(),
              "Additional op in region that should contain exactly one ABC_ExpressionOp.");
    return false;
  } else {
    return true;
  }
}

bool containsExactlyOneTargetNode(Region &region) {

  if (region.op_begin()==region.op_end()) {
    emitError(region.getLoc(), "Region must contain exactly one ABC_TargetOp but is empty.");
    return false;
  } else if (!region.op_begin()->hasTrait<OpTrait::isAbcTarget>()) {
    emitError(region.op_begin()->getLoc(), "Invalid op in region that should contain exactly one ABC_TargetOp.");
    return false;
  } else if (++region.op_begin()!=region.op_end()) {
    emitError((++region.op_begin())->getLoc(),
              "Additional op in region that should contain exactly one ABC_TargetOp.");
    return false;
  } else {
    return true;
  }
}


bool containsExactlyOneStatementNode(Region &region) {

  if (region.op_begin()==region.op_end()) {
    emitError(region.getLoc(), "Region must contain exactly one ABC_StatementOp but is empty.");
    return false;
  } else if (!region.op_begin()->hasTrait<OpTrait::isAbcStatement>()) {
    emitError(region.op_begin()->getLoc(), "Invalid op in region that should contain exactly one ABC_StatementOp.");
    return false;
  } else if (++region.op_begin()!=region.op_end()) {
    emitError((++region.op_begin())->getLoc(),
              "Additional op in region that should contain exactly one ABC_StatementOp.");
    return false;
  } else {
    return true;
  }
}

#include "ABC/ABCOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ABC dialect.
//===----------------------------------------------------------------------===//

void ABCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ABC/ABCOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ABC/ABCOps.cpp.inc"