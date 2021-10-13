#include "Batching.h"

#include <iostream>
#include <memory>
#include <unordered_map>

using namespace mlir;
using namespace abc;

int getConstIndex(tensor::ExtractOp &op) {
  //TODO: check if there's more and throw an error if yes
  auto index = *op.indices().begin();
  if (auto const_op = llvm::dyn_cast<ConstantOp>(index.getDefiningOp())) {
    return const_op.value().dyn_cast<IntegerAttr>().getValue().getLimitedValue();
  } else {
    emitError(index.getLoc(), "Index not defined by a constant op!");
  }
}

int getConstIndex(tensor::InsertOp &op) {
  //TODO: check if there's more and throw an error if yes
  auto index = *op.indices().begin();
  if (auto const_op = llvm::dyn_cast<ConstantOp>(index.getDefiningOp())) {
    return const_op.value().dyn_cast<IntegerAttr>().getValue().getLimitedValue();
  } else {
    emitError(index.getLoc(), "Index not defined by a constant op!");
  }
}

void resolveToSlot(int slot, Value v, IRRewriter &rewriter) {
  //TODO: This is the naive/dumb way, no checking if stuff has already happened!
  auto op = v.getDefiningOp();

  if (op) {
    if (auto extract_op = llvm::dyn_cast<tensor::ExtractOp>(op)) {
      // if it's an extract op, simply rotate to bring to right slot
      int extracted_index = getConstIndex(extract_op);
      int offset = extracted_index - slot;
      rewriter.setInsertionPointAfter(extract_op);
      auto offset_val = rewriter.create<ConstantOp>(extract_op.getLoc(), rewriter.getIndexAttr(offset));
      auto new_val = rewriter.create<abc::RotateOp>(extract_op.getLoc(),
                                                    extract_op.tensor().getType(),
                                                    extract_op.tensor(),
                                                    offset_val);

      extract_op.result().replaceAllUsesWith(new_val);
    } else {
      // any other operation just gets resolved recursively
      for (auto operand: op->getOperands()) {
        resolveToSlot(slot, operand, rewriter);
      }
    }
  } else {
    //ignore, not defined in this part of code (e.g., might be a function param)
  }
}

void BatchingPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalOp<AffineForOp>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  /// Struct for batching info
  struct batching_info {
    int slot;
    std::shared_ptr<Value> value;
  };

  /// Lookup stuff
  std::unordered_map<size_t, batching_info> slot_map;



  //TODO: There's very likely a much better way to do this that's not this kinf of manual walk!

  // First, read all the tensor.extract info to find the slot for each SSA value
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<tensor::ExtractOp>())) {
      int index_int = getConstIndex(op);
      auto val_ptr = std::make_shared<Value>(op.tensor());
      batching_info bi = {index_int, val_ptr};
      size_t hash = hash_value(op.result());
      slot_map.insert({hash, bi});
    }
  }

  // Now visit each InsertOp and translate it (if necessary)
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<tensor::InsertOp>())) {

      int target_index_int = getConstIndex(op);
      resolveToSlot(target_index_int, op.dest(), rewriter);

      //TODO: Replace the result of this insert op with the new_value

    }
  }

}