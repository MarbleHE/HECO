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

void collapseSub(abc::SubOp &op, IRRewriter &rewriter) {
  for (auto &use: llvm::make_early_inc_range(op.output().getUses())) {
    if (auto use_sub = llvm::dyn_cast<abc::SubOp>(*use.getOwner())) {
      rewriter.setInsertionPointAfter(use_sub.getOperation());
      llvm::SmallVector<Value, 4> new_summands;
      for (auto s: use_sub.summand()) {
        if (s!=op.output()) {
          new_summands.push_back(s);
        }
      }
      for (auto s: op.summand()) {
        new_summands.push_back(s);
      }
      auto new_sub = rewriter.create<abc::SubOp>(use_sub->getLoc(), use_sub.output().getType(), new_summands);
      use_sub.replaceAllUsesWith(new_sub.getOperation());
    }
  }
}

void collapseMul(abc::MulOp &op, IRRewriter &rewriter) {
  for (auto &use: llvm::make_early_inc_range(op.output().getUses())) {
    if (auto use_mul = llvm::dyn_cast<abc::MulOp>(*use.getOwner())) {
      rewriter.setInsertionPointAfter(use_mul.getOperation());
      llvm::SmallVector<Value, 4> new_summands;
      for (auto s: use_mul.terms()) {
        if (s!=op.output()) {
          new_summands.push_back(s);
        }
      }
      for (auto s: op.terms()) {
        new_summands.push_back(s);
      }
      auto new_mul = rewriter.create<abc::MulOp>(use_mul->getLoc(), use_mul.output().getType(), new_summands);
      use_mul.replaceAllUsesWith(new_mul.getOperation());
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
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<AddIOp>())) {
      rewriter.setInsertionPointAfter(op.getOperation());
      llvm::SmallVector<Value, 2> summands = {op.lhs(), op.rhs()};
      Value value = rewriter.create<abc::AddOp>(op.getLoc(), op.result().getType(), summands);
      op.replaceAllUsesWith(value);
      rewriter.eraseOp(op.getOperation());

    }
  }

  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<SubIOp>())) {
      rewriter.setInsertionPointAfter(op.getOperation());
      llvm::SmallVector<Value, 2> summands = {op.lhs(), op.rhs()};
      Value value = rewriter.create<abc::SubOp>(op.getLoc(), op.result().getType(), summands);
      op.replaceAllUsesWith(value);
      rewriter.eraseOp(op.getOperation());

    }
  }

  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<MulIOp>())) {
      rewriter.setInsertionPointAfter(op.getOperation());
      llvm::SmallVector<Value, 2> summands = {op.lhs(), op.rhs()};
      Value value = rewriter.create<abc::MulOp>(op.getLoc(), op.result().getType(), summands);
      op.replaceAllUsesWith(value);
      rewriter.eraseOp(op.getOperation());
    }
  }

  // Now, go through and actually start collapsing the operations!
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<abc::AddOp>())) {
      collapseAdd(op, rewriter);
    }
  }

  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<abc::SubOp>())) {
      collapseSub(op, rewriter);
    }
  }

  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<abc::MulOp>())) {
      collapseMul(op, rewriter);
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

//TODO: NOTE WE MUST ADD CSE MANUALLY!