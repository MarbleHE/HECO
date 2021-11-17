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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <iostream>

using namespace mlir;
using namespace fhe;

namespace {
#include "LowerFHEtoPoly.inc"
}

struct LowerFHEtoPolyPass : public mlir::PassWrapper<LowerFHEtoPolyPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect, FHEDialect>();
  }
  void runOnOperation() override {

    std::cout << "RUNNING" << std::endl;

    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, StandardOpsDialect, FHEDialect>();
    target.addIllegalOp<MultiplyOp>();

    mlir::RewritePatternSet patterns(&getContext());

    //TODO: ADD NEWLY ADDED REWRITE PATTERNS HERE
    patterns.add<SwitchMultPattern>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  };

  mlir::StringRef getArgument() const final {
    return "fhe2poly";
  }
};

int main(int argc, char **argv) {
  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<FHEDialect>();
  context.loadDialect<FHEDialect>();

  PassRegistration<LowerFHEtoPolyPass>();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "FHE optimizer driver\n", registry));
}
