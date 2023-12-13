#include "heco/Passes/evarelinearize/InsertRelinearize.h"
#include <iostream>
#include "heco/IR/EVA/EVADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace heco;

void InsertRelinearizePass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<eva::EVADialect>();
}

/// Basic Pattern for operations without attributes.
class RelinearizePattern final : public RewritePattern
{
public:
RelinearizePattern(PatternBenefit benefit, mlir::MLIRContext *context) :
    RewritePattern(eva::MultiplyOp::getOperationName(), benefit, context) {}

    LogicalResult matchAndRewrite(
        Operation *op, PatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        assert(op->getNumOperands() == 2);

        llvm::SmallVector<mlir::Value, 2> operands;
        for (auto operand : op->getOperands())
        {
            operands.push_back(operand);
        }
        
        Operation *relinearize = rewriter.replaceOpWithNewOp<eva::RelinearizeOp>(op, TypeRange(op->getResultTypes()), operands[0]);
        

        rewriter.setInsertionPoint(relinearize);
        Operation *new_mult = rewriter.create<eva::MultiplyOp>(op->getLoc(), relinearize->getResultTypes(), operands);
        
        relinearize->setOperand(0, new_mult->getResult(0));
        return success();
    };
};

void InsertRelinearizePass::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RelinearizePattern>(PatternBenefit(10), patterns.getContext());

    GreedyRewriteConfig config;
    // force topological order of processing operations
    config.useTopDownTraversal = true;
    config.maxIterations = 1;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)))
        signalPassFailure();
    
    getOperation()->print(llvm::outs());
}