#include "Nary.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace abc;

void collapseAdd(abc::AddOp &op, IRRewriter &rewriter) {
  for (auto &use: llvm::make_early_inc_range(op.output().getUses())) {
    if (auto use_add = llvm::dyn_cast<abc::AddOp>(*use.getOwner())) {
      rewriter.setInsertionPointAfter(use_add.getOperation());
      llvm::SmallVector<Value, 4> new_summands;
      for (auto s: use_add.summand()) {
        if (s!=op.output()) {
          new_summands.push_back(s);
        }
      }
      for (auto s: op.summand()) {
        new_summands.push_back(s);
      }
      auto new_add = rewriter.create<abc::AddOp>(use_add->getLoc(), use_add.output().getType(), new_summands);
      use_add.replaceAllUsesWith(new_add.getOperation());
    }
  }
}

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

  // Now, go through and actually start collapsing the operations!
  // TODO: Do the same for std.muli and abc.fhe_mul
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<abc::AddOp>())) {
      collapseAdd(op, rewriter);
    }
  }

  // TODO: Finally, clean up ops without uses (hard to do during above loop because of iter invalidation)
  //  HACK: just canonicalize again for now ;)
//  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
//    auto range = iterator_range<Block::reverse_iterator>(f.body().getBlocks().begin()->rbegin(),
//                                                         f.body().getBlocks().begin()->rend());
//    for (auto &op: llvm::make_early_inc_range(range)) {
//      if (auto add_op = llvm::dyn_cast<AddIOp>(op)) {
//        if (add_op->getUsers().empty()) {
//          rewriter.eraseOp(add_op.getOperation());
//        }
//      }
//    }
//  }
}