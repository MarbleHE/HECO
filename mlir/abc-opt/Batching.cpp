#include "Batching.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace abc;

void BatchingPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalOp<AffineForOp>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  //TODO: There's very likely a much better way to do this that's not this kinf of manual walk!
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto &op: llvm::make_early_inc_range(f.body().getOps())) {
      //TODO: Do stuff to the operations!
    }
  }

}