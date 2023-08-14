#include "heco/Passes/hir2hir/UnrollLoops.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

void unrollLoop(affine::AffineForOp &op, IRRewriter &rewriter)
{
    // First, let's recursively unroll all nested loops:
    for (auto nested_loop : op.getOps<affine::AffineForOp>())
    {
        unrollLoop(nested_loop, rewriter);
    }

    // TODO: Fix MLIR issues in mlir/Transforms/LoopUtils.h where the typedef for FuncOp is messing with the fwd
    // declaration of FuncOp
    if (loopUnrollFull(op).failed())
    {
        emitError(op.getLoc(), "Failed to unroll loop");
    }
}

void UnrollLoopsPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalOp<affine::AffineForOp>();

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    // TODO: There's likely a much better way to do this that's not this kind of manual walk!
    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<affine::AffineForOp>()))
        {
            unrollLoop(op, rewriter);
        }
    }
}