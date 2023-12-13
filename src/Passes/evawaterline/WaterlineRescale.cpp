#include "heco/Passes/evawaterline/WaterlineRescale.h"
#include <iostream>
#include "heco/IR/EVA/EVADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace heco;
using namespace eva;

void WaterlineRescalePass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<eva::EVADialect>();
}

// Insert rescale operations only after multiplications,
// because only they can increase the scale
class RescalePattern final : public RewritePattern
{
public:
RescalePattern(PatternBenefit benefit, mlir::MLIRContext *context, eva::ScaleAnalysis *_analysis) :
    RewritePattern(eva::MultiplyOp::getOperationName(), benefit, context) {
        analysis = _analysis;
    }

    LogicalResult matchAndRewrite(
        Operation *op, PatternRewriter &rewriter) const override
    {    
        assert(op->getNumOperands() == 2);
        
        int num_rescales = analysis->recommendRescale(op);
        assert("Recommendation missing" && num_rescales >= 0);

        while (num_rescales--) {

            llvm::SmallVector<mlir::Value, 2> operands;
            for (auto operand : op->getOperands())
            {
                operands.push_back(operand);
            }

            rewriter.setInsertionPoint(op);
            Operation *rescale = rewriter.replaceOpWithNewOp<eva::RescaleOp>(op, TypeRange(op->getResultTypes()), operands[0]);
            

            rewriter.setInsertionPoint(rescale);
            Operation *new_mult = rewriter.create<eva::MultiplyOp>(op->getLoc(), rescale->getResultTypes(), operands);
            
            rescale->setOperand(0, new_mult->getResult(0));
            op = new_mult;
        }

        return success();
    };
    
    private:
        eva::ScaleAnalysis *analysis;
};

void WaterlineRescalePass::runOnOperation()
{
    // TODO: We still need to emit a pre-amble with an include statement
    //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

    mlir::RewritePatternSet patterns(&getContext());

    // Configure the ScaleAnalysis
    eva::ScaleAnalysis::argument_scale = source_scale;
    eva::ScaleAnalysis::waterline = waterline;
    eva::ScaleAnalysis::scale_drop = scale_drop;

    ScaleAnalysis analysis = getAnalysis<ScaleAnalysis>();

    patterns.add<RescalePattern>(PatternBenefit(10), patterns.getContext(), &analysis);

    GreedyRewriteConfig config;
    // force topological order of processing operations
    config.useTopDownTraversal = true;
    config.maxIterations = 1;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)))
        signalPassFailure();

    getOperation()->print(llvm::outs());
}