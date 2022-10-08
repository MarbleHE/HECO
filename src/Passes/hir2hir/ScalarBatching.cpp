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

template <typename OpType>
LogicalResult scalarBatchArithmeticOperation(IRRewriter &rewriter, MLIRContext *context, OpType op)
{
    // TODO: Implement Scalar Batching Pass
    return success();
}

void ScalarBatchingPass::runOnOperation()
{
    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    // TODO: Some kind of datastructure to maintain information between operations

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        if (f.walk([&](Operation *op) {
                 // op->print(llvm::outs());
                 // llvm::outs() << "\n";
                 // llvm::outs().flush();
                 if (auto sub_op = llvm::dyn_cast_or_null<fhe::SubOp>(op))
                 {
                     if (scalarBatchArithmeticOperation<fhe::SubOp>(rewriter, &getContext(), sub_op).failed())
                         return WalkResult::interrupt();
                 }
                 else if (auto add_op = llvm::dyn_cast_or_null<fhe::AddOp>(op))
                 {
                     if (scalarBatchArithmeticOperation<fhe::AddOp>(rewriter, &getContext(), add_op).failed())
                         return WalkResult::interrupt();
                 }
                 else if (auto mul_op = llvm::dyn_cast_or_null<fhe::MultiplyOp>(op))
                 {
                     if (scalarBatchArithmeticOperation<fhe::MultiplyOp>(rewriter, &getContext(), mul_op).failed())
                         return WalkResult::interrupt();
                 }
                 // TODO: Add support for relinearization!
                 return WalkResult(success());
             }).wasInterrupted())
            signalPassFailure();
    }
}