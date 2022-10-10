#include "heco/Passes/hir2hir/LowerVirtual.h"
#include <queue>
#include <unordered_map>
#include "heco/IR/FHE/FHEDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void LowerVirtualPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<fhe::FHEDialect>();
}

LogicalResult lowerExtractOperation(IRRewriter &rewriter, MLIRContext *context, fhe::ExtractOp op)
{
    // TODO: Implement
    return success();
}

void LowerVirtualPass::runOnOperation()
{
    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if (f.walk([&](Operation *op) {
                 // llvm::outs() << "Walking operation ";
                 // op->print(llvm::outs());
                 // llvm::outs() << "\n";
                 // llvm::outs().flush();
                 if (auto sub_op = llvm::dyn_cast_or_null<fhe::ExtractOp>(op))
                 {
                     if (lowerExtractOperation(rewriter, &getContext(), sub_op).failed())
                         return WalkResult::interrupt();
                 }
                 // else if (auto add_op = llvm::dyn_cast_or_null<fhe::InsertOp>(op))
                 // {
                 //     if (lowerInsertOperation(rewriter, &getContext(), add_op, map).failed())
                 //         return WalkResult::interrupt();
                 // }
                 // else // if (auto mul_op = llvm::dyn_cast_or_null<fhe::MultiplyOp>(op))
                 // {
                 //     // TODO: ADD SUPPORT FOR COMBINE OP
                 //     // if (batchArithmeticOperation<fhe::MultiplyOp>(rewriter, &getContext(), mul_op, map).failed())
                 //     //     return WalkResult::interrupt();
                 // }
                 // TODO: Add support for relinearization!
                 return WalkResult(success());
             }).wasInterrupted())
            signalPassFailure();
    }
}