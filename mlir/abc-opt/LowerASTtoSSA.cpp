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

struct ReturnOpLowering : public mlir::ConversionPattern {
  ReturnOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(abc::ReturnOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    std::cout << "WOHOO" << std::endl;

    // TODO: Fix this, for now replace it with an mlir::ReturnOp, ignoring value
    rewriter.create<mlir::ReturnOp>(loc);

    return success();
  }
};




void LowerASTtoSSAPass::runOnOperation() {
  std::cout << "RUN ON OP" << std::endl;
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  target.addLegalOp<mlir::ReturnOp>();
  // target.addIllegalDialect<ABCDialect>();
  target.addIllegalOp<abc::ReturnOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<ReturnOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

///// Create a pass for lowering operations in the `Affine` and `Std` dialects,
///// for a subset of the Toy IR (e.g. matmul).
//std::unique_ptr<Pass> abc::createLowerASTtoSSAPass() {
//  std::cout << "CREATED A PASS" << std::endl;
//  return std::make_unique<LowerASTtoSSAPass>();
//}