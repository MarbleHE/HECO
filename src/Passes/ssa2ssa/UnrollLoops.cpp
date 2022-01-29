#include <iostream>
#include <memory>
#include "mlir/Transforms/LoopUtils.h"
#include "abc/Passes/ssa2ssa/UnrollLoops.h"

using namespace mlir;

void unrollLoop(AffineForOp& op, IRRewriter& rewriter) {

  // First, let's recursively unroll all nested loops:
  for(auto nested_loop : op.getOps<AffineForOp>()) {
    unrollLoop(nested_loop, rewriter);
  }

  if(loopUnrollFull(op).failed()) {
    emitError(op.getLoc(), "Failed to unroll loop");
  }
}

void UnrollLoopsPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalOp<AffineForOp>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  //TODO: There's likely a much better way to do this that's not this kind of manual walk!
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<AffineForOp>())) {
        unrollLoop(op, rewriter);
    }
  }

}