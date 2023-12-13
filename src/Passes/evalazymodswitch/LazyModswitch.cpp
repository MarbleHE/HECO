#include "heco/Passes/evalazymodswitch/LazyModswitch.h"
#include <iostream>
#include "heco/IR/EVA/EVADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace heco;
using namespace eva;

void LazyModswitchPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<eva::EVADialect>();
}

template <typename OpTy>
class LazyModswitchPattern final : public RewritePattern
{
public:
LazyModswitchPattern(PatternBenefit benefit, mlir::MLIRContext *context, eva::ModuloAnalysis *_analysis) :
    RewritePattern(OpTy::getOperationName(), benefit, context) {
        analysis = _analysis;
    }

    LogicalResult matchAndRewrite(
        Operation *op, PatternRewriter &rewriter) const override
    {
        std::string op_name = op->getName().getStringRef().str();

        int left_modulo = analysis->getValueModuloChainLength(op->getOperand(0));
        int right_modulo = analysis->getValueModuloChainLength(op->getOperand(1));
        
        if (left_modulo == -1 or right_modulo == -1) {
            return failure();
        }
    
        if (left_modulo == right_modulo) {
            return failure();
        }

        rewriter.setInsertionPoint(op);        
        
        while (left_modulo > right_modulo) {

            Operation *modswitch = rewriter.create<eva::ModSwitchOp>(op->getLoc(), op->getOperand(0).getType(), op->getOperand(0));
            op = rewriter.replaceOpWithNewOp<OpTy>(op, TypeRange(op->getResultTypes()), ValueRange({modswitch->getResult(0), op->getOperand(1)}));
            left_modulo -= 1;
        }

        while (right_modulo > left_modulo) {

            Operation *modswitch = rewriter.create<eva::ModSwitchOp>(op->getLoc(), op->getOperand(1).getType(), op->getOperand(1));
            op = rewriter.replaceOpWithNewOp<OpTy>(op, TypeRange(op->getResultTypes()), ValueRange({op->getOperand(0), modswitch->getResult(0)}));
            right_modulo -= 1;
        }

        return success();
    };
    
    private:
        eva::ModuloAnalysis *analysis;
};

void LazyModswitchPass::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());

    // Configure the ScaleAnalysis
    eva::ModuloAnalysis::argument_modulo = source_modulo;

    ModuloAnalysis analysis = getAnalysis<ModuloAnalysis>();

    patterns.add<LazyModswitchPattern<eva::AddOp>>(PatternBenefit(10), patterns.getContext(), &analysis);
    patterns.add<LazyModswitchPattern<eva::SubOp>>(PatternBenefit(10), patterns.getContext(), &analysis);
    patterns.add<LazyModswitchPattern<eva::MultiplyOp>>(PatternBenefit(10), patterns.getContext(), &analysis);

    GreedyRewriteConfig config;
    // force topological order of processing operations
    config.useTopDownTraversal = true;
    config.maxIterations = 1;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)))
        signalPassFailure();

    getOperation()->print(llvm::outs());
}