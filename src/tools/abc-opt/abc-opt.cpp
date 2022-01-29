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

void pipelineBuilder(OpPassManager &manager) {
  manager.addPass(std::make_unique<LowerASTtoSSAPass>());
  manager.addPass(std::make_unique<UnrollLoopsPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass()); //this can greatly reduce the number of operations after unrolling
  manager.addPass(std::make_unique<NaryPass>());

  // Must canonicalize before Tensor2BatchedSecretPass, since it only handles constant indices in tensor.extract
  manager.addPass(createCanonicalizerPass());
  manager.addPass(std::make_unique<Tensor2BatchedSecretPass>());
  manager.addPass(createCanonicalizerPass()); //necessary to remove redundant fhe.materialize
  manager.addPass(createCSEPass()); //necessary to remove duplicate fhe.extract

  manager.addPass(std::make_unique<BatchingPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass()); // otherwise, the internal batching pass has no "same origin" things to find!
  manager.addPass(createCanonicalizerPass()); // to fold combine ops that might have simpler form after CSE

  manager.addPass(std::make_unique<InternalOperandBatchingPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());

  manager.addPass(std::make_unique<LowerToEmitCPass>());
  manager.addPass(createCanonicalizerPass()); //necessary to remove redundant fhe.materialize
}

void ssaPipelineBuilder(OpPassManager &manager) {
  manager.addPass(std::make_unique<UnrollLoopsPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass()); //this can greatly reduce the number of operations after unrolling
  manager.addPass(std::make_unique<NaryPass>());

  // Must canonicalize before Tensor2BatchedSecretPass, since it only handles constant indices in tensor.extract
  manager.addPass(createCanonicalizerPass());
  manager.addPass(std::make_unique<Tensor2BatchedSecretPass>());
  manager.addPass(createCanonicalizerPass()); //necessary to remove redundant fhe.materialize
  manager.addPass(createCSEPass()); //necessary to remove duplicate fhe.extract

  manager.addPass(std::make_unique<BatchingPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass()); // otherwise, the internal batching pass has no "same origin" things to find!
  manager.addPass(createCanonicalizerPass()); // to fold combine ops that might have simpler form after CSE

  manager.addPass(std::make_unique<InternalOperandBatchingPass>());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());

  manager.addPass(std::make_unique<LowerToEmitCPass>());
  manager.addPass(createCanonicalizerPass()); //necessary to remove redundant fhe.materialize
}

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<ABCDialect>();
  registry.insert<FHEDialect>();
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<arith::ArithmeticDialect>();
  context.loadDialect<ABCDialect>();
  context.loadDialect<FHEDialect>();
  context.loadDialect<AffineDialect>();
  context.loadDialect<tensor::TensorDialect>();
  context.loadDialect<arith::ArithmeticDialect>();
  // Uncomment the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  // Uncomment the following to make *all* MLIR core passes available.
  // This is only useful for experimenting with the command line to compose
  //registerAllPasses();

  registerCanonicalizerPass();
  registerAffineLoopUnrollPass();
  registerCSEPass();
  PassRegistration<LowerASTtoSSAPass>();
  PassRegistration<UnrollLoopsPass>();
  PassRegistration<NaryPass>();
  PassRegistration<Tensor2BatchedSecretPass>();
  PassRegistration<BatchingPass>();
  PassRegistration<InternalOperandBatchingPass>();
  PassRegistration<LowerToEmitCPass>();

  PassPipelineRegistration<>("full-pass", "Run all passes", pipelineBuilder);
  PassPipelineRegistration<>("from-ssa-pass", "Run all passes starting with ssa", ssaPipelineBuilder);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "ABC optimizer driver\n", registry));
}
