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
//
//struct ReturnOpLowering : public mlir::ConversionPattern {
//  ReturnOpLowering(mlir::MLIRContext *ctx)
//      : mlir::ConversionPattern(abc::ReturnOp::getOperationName(), 1, ctx) {}
//
//  mlir::LogicalResult
//  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
//                  mlir::ConversionPatternRewriter &rewriter) const final {
//
//    // TODO: For now replace it with an mlir::ReturnOp, ignoring value
//    // rewriter.create<mlir::ReturnOp>(loc);
//    rewriter.eraseOp(op);
//
//    return success();
//  }
//};
//
//struct FunctionOpLowering : public mlir::ConversionPattern {
//  FunctionOpLowering(mlir::MLIRContext *ctx)
//      : mlir::ConversionPattern(abc::FunctionOp::getOperationName(), 1, ctx) {}
//
//  mlir::LogicalResult
//  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
//                  mlir::ConversionPatternRewriter &rewriter) const final {
//    auto loc = op->getLoc();
//
//    auto name = op->getAttr("name").cast<StringAttr>().getValue();
//    loc.dump();
//    std::cout << name.str() << std::endl;
//
//    // TODO: Get the arguments and their types from the first region
//    // TODO: Get real return type (convert from string or change IR to have real types)
//    auto type = rewriter.getFunctionType(llvm::None, llvm::None);
//
//
//
//    // // Get the block from the second region
//    // auto &body_region = op->getRegion(1);
//    // auto &body_block  = body_region.getBlocks().front();
//
//    rewriter.eraseOp(op);
//    // auto f = rewriter.create<mlir::FuncOp>(loc, name, type);
//    //std::cout << "F:" << std::endl;
//    //f.dump();
//    //std::cout << "/F" << std::endl;
//
//    return success();
//  }
//};

mlir::FuncOp convertFunctionOp2FuncOp(FunctionOp f) {
  auto new_f = mlir::FuncOp::create(f.getLoc(), "test", FunctionType());
  std::cout << "NEW FUNCTION:" << std::endl;
  new_f.dump();
  return new_f;
}

void LowerASTtoSSAPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  target.addLegalOp<mlir::ReturnOp>();
  // target.addIllegalDialect<ABCDialect>();
  target.addIllegalOp<abc::ReturnOp>();

  auto &block = getOperation()->getRegion(0).getBlocks().front();

  for (auto &f: llvm::make_early_inc_range(block)) {
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(&f);
    auto new_f = rewriter.create<FuncOp>(f.getLoc(), "testing", rewriter.getFunctionType(llvm::None, llvm::None));
    new_f.setPrivate();
    rewriter.eraseOp(&f);
  }

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

///// Create a pass for lowering operations in the `Affine` and `Std` dialects,
///// for a subset of the Toy IR (e.g. matmul).
//std::unique_ptr<Pass> abc::createLowerASTtoSSAPass() {
//  std::cout << "CREATED A PASS" << std::endl;
//  return std::make_unique<LowerASTtoSSAPass>();
//}