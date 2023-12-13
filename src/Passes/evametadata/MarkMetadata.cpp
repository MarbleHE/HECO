#include "heco/Passes/evametadata/MarkMetadata.h"
#include <iostream>
#include "heco/IR/EVA/EVADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace heco;
using namespace eva;

void MarkMetadataPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<eva::EVADialect>();
}

template <typename OpTy>
class MarkMetadataPattern final : public RewritePattern
{
public:
MarkMetadataPattern(PatternBenefit benefit, mlir::MLIRContext *context, eva::ScaleAnalysis *_scale_analysis, eva::ModuloAnalysis *_modulo_analysis) :
    RewritePattern(OpTy::getOperationName(), benefit, context) {
        scale_analysis = _scale_analysis;
        modulo_analysis = _modulo_analysis;
    }

    LogicalResult matchAndRewrite(
        Operation *op, PatternRewriter &rewriter) const override
    {
        int scale = scale_analysis->getScaleWithoutImpliedRescales(op);
        int modulo = modulo_analysis->getModuloChainLength(op);

        auto specific_op = dyn_cast<OpTy>(op);
        specific_op.setResultScale(scale);
        specific_op.setResultMod(modulo);
        return success();
    };
    
    private:
        eva::ScaleAnalysis *scale_analysis;
        eva::ModuloAnalysis *modulo_analysis;
};

void MarkMetadataPass::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());

    // Configure the ScaleAnalysis
    eva::ScaleAnalysis::argument_scale = source_scale;
    eva::ScaleAnalysis::waterline = waterline;
    eva::ScaleAnalysis::scale_drop = scale_drop;
    eva::ModuloAnalysis::argument_modulo = source_modulo;

    ScaleAnalysis scale_analysis = getAnalysis<ScaleAnalysis>();
    ModuloAnalysis modulo_analysis = getAnalysis<ModuloAnalysis>();

    patterns.add<MarkMetadataPattern<eva::NegateOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::MultiplyOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::AddOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::SubOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::RelinearizeOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::RescaleOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::ModSwitchOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::ConstOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);
    patterns.add<MarkMetadataPattern<eva::RotateOp>>(PatternBenefit(10), patterns.getContext(), &scale_analysis, &modulo_analysis);

    GreedyRewriteConfig config;
    // force topological order of processing operations
    config.useTopDownTraversal = true;
    config.maxIterations = 1;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)))
        signalPassFailure();

    getOperation()->print(llvm::outs());
}