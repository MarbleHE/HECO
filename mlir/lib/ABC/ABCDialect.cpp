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