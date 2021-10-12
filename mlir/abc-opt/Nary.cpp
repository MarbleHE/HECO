#include "Nary.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace abc;

void NaryPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalOp<AffineForOp>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  //TODO: There's very likely a much better way to do this that's not this kind of manual walk!

  // First, go and translate std.addi to abc.fhe_add
  // TODO: Do the same for std.muli and abc.fhe_mul
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<AddIOp>())) {
      rewriter.setInsertionPointAfter(op.getOperation());
      llvm::SmallVector<Value, 2> summands = {op.lhs(), op.rhs()};
      Value value = rewriter.create<abc::AddOp>(op.getLoc(), op.result().getType(), summands);
      op.replaceAllUsesWith(value);
      rewriter.eraseOp(op.getOperation());

    }
  }

}