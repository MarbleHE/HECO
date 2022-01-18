
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "abc/IR/FHE/FHEDialect.h"
#include "abc/Passes/ssa2ssa/InternalOperandBatching.h"

using namespace mlir;

void InternalOperandBatchingPass::getDependentDialects(mlir::DialectRegistry &registry) const {
  registry.insert<fhe::FHEDialect,
                  mlir::AffineDialect,
                  mlir::StandardOpsDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}

template<typename OpType>
LogicalResult internalBatchArithmeticOperation(IRRewriter &rewriter, MLIRContext *context, OpType op) {
  // We care only about ops that return batched values, which should be pretty much all fhe ops after the "batching" pass
  if (auto result_bst = op.getType().template dyn_cast_or_null<fhe::BatchedSecretType>()) {

    //llvm::outs() << "updating ";
    //op.print(llvm::outs());
    //llvm::outs() << "\n";


    /// Target Slot (-1 => no target slot required)
    int target_slot = -1;

    /// Helper Struct for uses
    struct BatchingUse {
      int index = -1;
      Value occurrence;
      BatchingUse(int index, Value occurrence) : index(index), occurrence(occurrence) {};
    };

    /// List of used (Batched type) operand inputs' origins and the indices accessed in each one
    typedef llvm::SmallMapVector<Value, std::vector<BatchingUse>, 1> OriginMap;
    OriginMap originMap;

    auto addOriginUse = [&](Value o, int index, Value occurrence) {
      if (originMap.find(o)!=originMap.end())
        originMap.find(o)->second.push_back(BatchingUse(index, occurrence));
      else
        originMap.insert({o, {BatchingUse(index, occurrence)}});
    };



    // collect origin information
    for (auto it = op->operand_begin(); it!=op.operand_end(); ++it) {

      if (auto bst = (*it).getType().template dyn_cast_or_null<fhe::BatchedSecretType>()) {
        if (auto r_op = (*it).template getDefiningOp<fhe::RotateOp>()) {
          addOriginUse(r_op.x(), -r_op.i(), *it);
        } else if (auto c_op = (*it).template getDefiningOp<fhe::ConstOp>()) {
          // Constant Ops can be realized to whatever slot we want them to be in
          addOriginUse(*it, -1, *it);
        } else {
          // Everything else is assumed to have been rotated to slot 0 by previous optimizations
          addOriginUse(*it, 0, *it);
        }
      } else {
        emitError(op.getLoc(),
                  "Encountered unexpected (non BatchedSecret) defining op as operand while trying to internally batch.");
        return failure();
      }

    }

    //for (auto el: originMap) {
    //  llvm::outs() << "uses of: ";
    //  el.first.print(llvm::outs());
    //  llvm::outs() << " : ";
    //  for (auto i: el.second) {
    //    llvm::outs() << i.index << ", ";
    //  }
    //  llvm::outs() << '\n';
    //}

    for (auto el: originMap) {
      auto origin = el.first;

      //llvm::outs() << "(potentially) collapsing uses of: ";
      //origin.print(llvm::outs());
      //llvm::outs() << '\n';

      auto uses = el.second;
      std::sort(uses.begin(), uses.end(), [](const BatchingUse &a, const BatchingUse &b) -> bool {
        return a.index < b.index;
      });

      //TODO: We eventually need to support arbitrary sets of indices, but for now we just do 0:(2^k) as a PoC
      bool contiguous = !uses.empty() && uses.size() > 1 && uses[0].index==0;
      for (size_t i = 1; i < uses.size(); ++i) {
        contiguous = contiguous && uses[i - 1].index + 1==uses[i].index;
      }

      if (contiguous) {
        //llvm::outs() << "trying to batch contiguous uses of : ";
        //el.first.print(llvm::outs());
        //llvm::outs() << '\n';

        // Check if it's a power of two
        auto n = uses[uses.size() - 1].index + 1;
        auto k = std::log2(n);
        if (k==(int) k) {
          rewriter.setInsertionPoint(op);
          Value prev = origin;
          Value added;
          for (int i = n/2; i > 0; i /= 2) {
            auto rotated_down = rewriter.template create<fhe::RotateOp>(op.getLoc(), prev, -i);
            added = rewriter.template create<fhe::AddOp>(op.getLoc(), ValueRange({prev, rotated_down}));
            prev = added;
          }

          // Now we need to replace ONE OF the operands that have this origin with "added" and REMOVE THE REST
          auto old_range = op.x();
          SmallVector<Value> new_range = {};
          for (auto v: old_range) {
            bool remove = false;
            for (auto u: uses) {
              if (v==u.occurrence)  // we need to remove this
                remove = true;
            }
            if (!remove)
              new_range.push_back(v);
          }
          new_range.push_back(added);

          //TODO: This is probably all kinds of unsafe if there are multiple origins that are being replaced in the same op
          op.xMutable().assign(new_range);

          //llvm::outs() << "current function: ";
          //op->getParentOp()->print(llvm::outs());
          //llvm::outs() << '\n';
        } else {
          //llvm::outs() << "Ignoring anyway because its not a power of two.\n";
        }

      } else {
        //llvm::outs() << "ignoring non-contiguous/non-repeated uses of : ";
        //el.first.print(llvm::outs());
        //llvm::outs() << '\n';
      }

    }

  }
  return
      success();
}

void InternalOperandBatchingPass::runOnOperation() {

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    // We must translate in order of appearance for this to work, so we walk manually
    if (f.walk([&](Operation *op) {
      if (auto sub_op = llvm::dyn_cast_or_null<fhe::SubOp>(op))
        if (internalBatchArithmeticOperation<fhe::SubOp>(rewriter, &getContext(), sub_op).failed())
          return WalkResult::interrupt();
      if (auto add_op = llvm::dyn_cast_or_null<fhe::AddOp>(op))
        if (internalBatchArithmeticOperation<fhe::AddOp>(rewriter, &getContext(), add_op).failed())
          return WalkResult::interrupt();
      if (auto mul_op = llvm::dyn_cast_or_null<fhe::MultiplyOp>(op))
        if (internalBatchArithmeticOperation<fhe::MultiplyOp>(rewriter, &getContext(), mul_op).failed())
          return WalkResult::interrupt();
      //TODO: Add support for relinearization!
      return WalkResult(success());
    }).wasInterrupted())
      signalPassFailure();
  }

}