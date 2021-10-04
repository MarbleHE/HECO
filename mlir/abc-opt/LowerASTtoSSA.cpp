//===----------------------------------------------------------------------===//
//
// This file implements a lowering of AST nodes in MLIR (ABC Dialect) to
// a combination of std, builtin, affine and sfc dialects in SSA form
//
//===----------------------------------------------------------------------===//


#include "LowerASTtoSSA.h"

#include <iostream>

using namespace mlir;
using namespace abc;

void convertFunctionOp2FuncOp(FunctionOp &f, IRRewriter &rewriter) {
  // TODO: Process the FunctionParameters

  // TODO: Get real return type (convert from string or change IR to have real types)


  rewriter.setInsertionPoint(f);
  auto new_f = rewriter.create<FuncOp>(f.getLoc(), f.name(), rewriter.getFunctionType(llvm::None, llvm::None));
  new_f.setPrivate();
  rewriter.eraseOp(f);
}

void LowerASTtoSSAPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  target.addLegalOp<mlir::ReturnOp>();
  // target.addIllegalDialect<ABCDialect>();

  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  for (auto f: llvm::make_early_inc_range(block.getOps<FunctionOp>())) {
    convertFunctionOp2FuncOp(f, rewriter);
  }

  // TODO: Lower the bodies of the FuncOPs, which are still ABC/AST

  // Next approach: Manually walking the IR

//  // Now that the conversion target has been defined, we just need to provide
//  // the set of patterns that will lower the Toy operations.
//  RewritePatternSet patterns(&getContext());
//  patterns.add<ReturnOpLowering>(&getContext());
//  patterns.add<FunctionOpLowering>(&getContext());

//  // With the target and rewrite patterns defined, we can now attempt the
//  // conversion. The conversion will signal failure if any of our `illegal`
//  // operations were not converted successfully.
//  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
//    signalPassFailure();
}