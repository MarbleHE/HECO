#include "Batching.h"

#include <iostream>
#include <memory>
#include <unordered_map>

using namespace mlir;
using namespace abc;

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
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    for (auto op: llvm::make_early_inc_range(f.body().getOps<tensor::ExtractOp>())) {
      //TODO: check if there's more and throw an error if yes
      auto index = *op.indices().begin();
      if (auto const_op = llvm::dyn_cast<ConstantOp>(index.getDefiningOp())) {
        int index_int = const_op.value().dyn_cast<IntegerAttr>().getValue().getLimitedValue();
        auto val_ptr = std::make_shared<Value>(op.tensor());
        batching_info bi = {index_int, val_ptr};
        size_t hash = hash_value(op.result());
        slot_map.insert({hash, bi});
      } else {
        emitError(index.getLoc(), "Index not defined by a constant op!");
      }

    }
  }

}