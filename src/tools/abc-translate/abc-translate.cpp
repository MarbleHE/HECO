//===- abc-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Translation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

#include "abc/IR/ABC/ABCDialect.h"
#include "abc/IR/FHE/FHEDialect.h"
#include "abc/Passes/ast2ssa/LowerASTtoSSA.h"
#include "abc/Passes/ssa2ssa/UnrollLoops.h"
#include "abc/Passes/ssa2ssa/Nary.h"
#include "abc/Passes/ssa2ssa/Tensor2BatchedSecret.h"
#include "abc/Passes/ssa2ssa/Batching.h"
#include "abc/Passes/ssa2ssa/InternalOperandBatching.h"
#include "abc/Passes/ssa2cpp/LowerToEmitC.h"

#include <iostream>

using namespace mlir;
using namespace abc;
using namespace fhe;


int main(int argc, char **argv) {
  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<ABCDialect>();
  registry.insert<FHEDialect>();
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<arith::ArithmeticDialect>();
  registry.insert<emitc::EmitCDialect>();
  context.loadDialect<ABCDialect>();
  context.loadDialect<FHEDialect>();
  context.loadDialect<AffineDialect>();
  context.loadDialect<tensor::TensorDialect>();
  context.loadDialect<arith::ArithmeticDialect>();
  context.loadDialect<emitc::EmitCDialect>();
  // Uncomment the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  registerToCppTranslation();


  return failed(
      mlir::mlirTranslateMain(argc, argv, "ABC Translation Tool"));

}