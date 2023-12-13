#include "heco/Passes/evamatchscale/MatchScale.h"
#include <iostream>
#include "heco/IR/EVA/EVADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace heco;
using namespace eva;

void MatchScalePass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<eva::EVADialect>();
}

template <typename OpTy>
class MatchScalePattern final : public RewritePattern
{
public:
MatchScalePattern(PatternBenefit benefit, mlir::MLIRContext *context, eva::ScaleAnalysis *_analysis) :
    RewritePattern(OpTy::getOperationName(), benefit, context) {
        analysis = _analysis;
    }

    LogicalResult matchAndRewrite(
        Operation *op, PatternRewriter &rewriter) const override
    {
        std::string op_name = op->getName().getStringRef().str();

        int left_scale = analysis->getValueScaleWithoutImpliedRescales(op->getOperand(0));
        int right_scale = analysis->getValueScaleWithoutImpliedRescales(op->getOperand(1));
        
        if (left_scale == -1 or right_scale == -1) {
            return failure();
        }
    
        if (left_scale == right_scale) {
            return failure();
        }

        rewriter.setInsertionPoint(op);
        
        int diff = std::abs(left_scale - right_scale);
        // we assume the lengths to be identical
        int vec_len = op->getOperand(0).getType().cast<CipherType>().getSize();
                
        if (left_scale < right_scale) {

            Operation *constant = rewriter.create<eva::ConstOp>(op->getLoc(), op->getOperand(0).getType(),
                    rewriter.getI32ArrayAttr(std::vector <int> (vec_len, 1)), diff, -1);

            Operation *mult = rewriter.create<eva::MultiplyOp>(op->getLoc(), op->getOperand(0).getType(), ValueRange({op->getOperand(0), constant->getResult(0)}));
            rewriter.replaceOpWithNewOp<OpTy>(op, TypeRange(op->getResultTypes()), ValueRange({mult->getResult(0), op->getOperand(1)}));

        } else {

            Operation *constant = rewriter.create<eva::ConstOp>(op->getLoc(), op->getOperand(0).getType(),
                    rewriter.getI32ArrayAttr(std::vector <int> (vec_len, 1)), diff, -1);

            Operation *mult = rewriter.create<eva::MultiplyOp>(op->getLoc(), op->getOperand(1).getType(), ValueRange({op->getOperand(1), constant->getResult(0)}));
            rewriter.replaceOpWithNewOp<OpTy>(op, TypeRange(op->getResultTypes()), ValueRange({op->getOperand(0), mult->getResult(0)}));
        }
        
        return success();
    };
    
    private:
        eva::ScaleAnalysis *analysis;
};

void MatchScalePass::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());

    // Configure the ScaleAnalysis
    eva::ScaleAnalysis::argument_scale = source_scale;
    eva::ScaleAnalysis::waterline = waterline;
    eva::ScaleAnalysis::scale_drop = scale_drop;

    ScaleAnalysis analysis = getAnalysis<ScaleAnalysis>();

    patterns.add<MatchScalePattern<eva::AddOp>>(PatternBenefit(10), patterns.getContext(), &analysis);
     patterns.add<MatchScalePattern<eva::SubOp>>(PatternBenefit(10), patterns.getContext(), &analysis);

    GreedyRewriteConfig config;
    // force topological order of processing operations
    config.useTopDownTraversal = true;
    config.maxIterations = 1;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)))
        signalPassFailure();

    getOperation()->print(llvm::outs());
}