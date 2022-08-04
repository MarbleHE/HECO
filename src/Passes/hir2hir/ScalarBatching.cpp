#include "heco/Passes/hir2hir/ScalarBatching.h"
#include "heco/IR/FHE/FHEDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void ScalarBatchingPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<
        fhe::FHEDialect, mlir::AffineDialect, func::FuncDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
}

void ScalarBatchingPass::runOnOperation()
{
    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if (f.walk([&](Operation *op) { return WalkResult(success()); }).wasInterrupted())
            signalPassFailure();
    }
}