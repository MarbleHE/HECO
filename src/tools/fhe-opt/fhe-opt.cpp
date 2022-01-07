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

#include "ast_opt/IR/FHE/FHEDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include <iostream>

using namespace mlir;
using namespace fhe;

auto make_const(Builder &builder, int64_t number) {
  return builder.getI64IntegerAttr(number);
}


static LogicalResult rewriteMultiply(fhe::MultiplyOp op, PatternRewriter &rewriter) {

  auto loc = op->getLoc();

  auto zero = rewriter.getI64IntegerAttr(0);
  auto one = rewriter.getI64IntegerAttr(1);
  auto x0 = rewriter.create<GetPolyOp>(loc, op.x().getType(), op.x(), zero);
  auto x1 = rewriter.create<GetPolyOp>(loc, op.x().getType(), op.x(), one);
  auto y0 = rewriter.create<GetPolyOp>(loc, op.y().getType(), op.y(), zero);
  auto y1 = rewriter.create<GetPolyOp>(loc, op.y().getType(), op.y(), one);

  auto t0 = rewriter.create<PolyMulOp>(loc, op.x().getType(), x0, y0, op.parms());
  auto temp = rewriter.create<PolyMulOp>(loc, op.x().getType(), x0, y1, op.parms());
  auto t1 = rewriter.create<PolyMulAccOp>(loc, op.x().getType(), x1, y0, temp, op.parms());
  auto t2 = rewriter.create<PolyMulOp>(loc, op.x().getType(), x1, y1, op.parms());
  rewriter.replaceOpWithNewOp<MakeCtxtOp>(op, op.x().getType(), t0, t1, t2);

  return success();
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

    // TableGen Version:
    // patterns.add<SwitchMultPattern>(&getContext());

    // C++ Version:
    patterns.add(rewriteMultiply);

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
