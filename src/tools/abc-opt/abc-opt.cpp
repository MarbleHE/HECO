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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

#include "ast_opt/IR/ABC/ABCDialect.h"
#include "ast_opt/Passes/ast2ssa/LowerASTtoSSA.h"
#include "UnrollLoops.h"
#include "Nary.h"
#include "Batching.h"

#include <iostream>

using namespace mlir;
using namespace abc;

void pipelineBuilder(OpPassManager &manager) {
  manager.addPass(std::make_unique<LowerASTtoSSAPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(std::make_unique<UnrollLoopsPass>());
  manager.addPass(createCanonicalizerPass());
  // manager.addPass(std::make_unique<BatchingPass>());
  // manager.addPass(createCanonicalizerPass());
  // manager.addPass(std::make_unique<NaryPass>());
  // manager.addPass(createCanonicalizerPass());
}

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<ABCDialect>();
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<arith::ArithmeticDialect>();
  context.loadDialect<ABCDialect>();
  context.loadDialect<AffineDialect>();
  context.loadDialect<tensor::TensorDialect>();
  context.loadDialect<arith::ArithmeticDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  registerAllPasses();
  PassRegistration<LowerASTtoSSAPass>();
  PassRegistration<UnrollLoopsPass>();
  PassRegistration<BatchingPass>();
  PassRegistration<NaryPass>();

  PassPipelineRegistration<>("full-pass", "Run all passes", pipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "ABC optimizer driver\n", registry));
}
