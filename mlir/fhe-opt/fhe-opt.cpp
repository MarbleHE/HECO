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

#include "FHE/FHEDialect.h"
#include "LowerFHEtoPoly.h"

#include <iostream>

using namespace mlir;
using namespace fhe;

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<FHEDialect>();
  registry.insert<StandardOpsDialect>();
  registry.insert<AffineDialect>();
  registry.insert<tensor::TensorDialect>();
  context.loadDialect<FHEDialect>();
  context.loadDialect<AffineDialect>();
  context.loadDialect<tensor::TensorDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  //  PassManager pm(&context);
  //  pm.addNestedPass<abc::ReturnOp>(abc::createLowerASTtoSSAPass());

  registerAllPasses();
  PassRegistration<LowerFHEtoPolyPass>();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "FHE optimizer driver\n", registry));
}
