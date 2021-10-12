#include "UnrollLoops.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace abc;

void unrollLoop(AffineForOp& op, IRRewriter& rewriter) {

  // First, let's recursively unroll all nested loops:
  for(auto nested_loop : op.getOps<AffineForOp>()) {
    unrollLoop(nested_loop, rewriter);
  }

  std::cout << "LOOKING AT: " << std::endl;
  op->dump();
  std::cout << "-------------" << std::endl;
}

void UnrollLoopsPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalOp<AffineForOp>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  //TODO: There's very likely a much better way to do this that's not this kinf of manual walk!
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<AffineForOp>())) {
        unrollLoop(op, rewriter);
    }
  }

}