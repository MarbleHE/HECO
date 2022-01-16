#include <iostream>
#include <memory>
#include <unordered_map>

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "abc/IR/FHE/FHEDialect.h"
#include "abc/Passes/ssa2ssa/Batching.h"

using namespace mlir;

void BatchingPass::getDependentDialects(mlir::DialectRegistry &registry) const {
  registry.insert<fhe::FHEDialect,
                  mlir::AffineDialect,
                  mlir::StandardOpsDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}

Value resolveToSlot(int slot, Value v, IRRewriter &rewriter, std::string indent = "") {
  //TODO: This is the naive/dumb way, no checking if stuff has already happened!

  llvm::outs() << indent << "resolving slot for: ";
  v.print(llvm::outs());
  llvm::outs() << indent << "\n";

  auto op = v.getDefiningOp();
  if (op) {
    if (auto extract_op = llvm::dyn_cast<fhe::ExtractOp>(op)) {
      // if it's an extract op, simply rotate to bring to right slot
      int extracted_index = (int) extract_op.i().getLimitedValue(INT32_MAX);
      int offset = extracted_index - slot;
      rewriter.setInsertionPointAfter(extract_op);
      auto new_val = rewriter.create<fhe::RotateOp>(extract_op.getLoc(),
                                                    extract_op.vector().getType(),
                                                    extract_op.vector(),
                                                    rewriter.getIndexAttr(offset));

      llvm::outs() << indent << "resolved extract to: ";
      new_val.print(llvm::outs());
      llvm::outs() << indent << "\n";

      return new_val;
    } else if (auto fhe_add = llvm::dyn_cast<fhe::AddOp>(op)) {
      llvm::SmallVector<Value, 4> new_summands;
      for (auto operand: fhe_add.x()) {
        auto new_val = resolveToSlot(slot, operand, rewriter, indent + "  ");
        new_summands.push_back(new_val);
      }
      // and we also need to update the return type for FHE ops.
      // TODO: would be great if we could do this generically!

      rewriter.setInsertionPointAfter(fhe_add);
      auto new_add = rewriter.create<fhe::AddOp>(fhe_add.getLoc(), new_summands.begin()->getType(), new_summands);
      fhe_add->replaceAllUsesWith(new_add);
      //TODO: THIS REQUIRES VERIFY-EACH-ZERO SO THE TYPE SYSTEM DOESN'T COMPLAIN TOO MUCH!

      auto new_val = *new_add->result_begin();
      llvm::outs() << indent << "resolved add to: ";
      new_val.print(llvm::outs());
      llvm::outs() << indent << "\n";

      return new_val;
    } else if (auto fhe_mul = llvm::dyn_cast<fhe::MultiplyOp>(op)) {
      llvm::SmallVector<Value, 4> new_summands;
      for (auto operand: fhe_mul.x()) {
        auto new_val = resolveToSlot(slot, operand, rewriter, indent + "  ");
        new_summands.push_back(new_val);
      }
      // and we also need to update the return type for FHE ops.
      // TODO: would be great if we could do this generically!

      rewriter.setInsertionPointAfter(fhe_mul);
      auto new_mul = rewriter.create<fhe::MultiplyOp>(fhe_mul.getLoc(), new_summands.begin()->getType(), new_summands);
      fhe_mul->replaceAllUsesWith(new_mul);
      //TODO: THIS REQUIRES VERIFY-EACH-ZERO SO THE TYPE SYSTEM DOESN'T COMPLAIN TOO MUCH!

      auto new_val = *new_mul->result_begin();

      llvm::outs() << indent << "resolved mul to: ";
      new_val.print(llvm::outs());
      llvm::outs() << indent << "\n";

      return new_val;
    } else if (auto fhe_sub = llvm::dyn_cast<fhe::SubOp>(op)) {
      llvm::SmallVector<Value, 4> new_summands;
      for (auto operand: fhe_sub.x()) {
        auto new_val = resolveToSlot(slot, operand, rewriter, indent + "  ");
        new_summands.push_back(new_val);
      }
      // and we also need to update the return type for FHE ops.
      // TODO: would be great if we could do this generically!

      rewriter.setInsertionPointAfter(fhe_sub);
      auto new_sub = rewriter.create<fhe::SubOp>(fhe_sub.getLoc(), new_summands.begin()->getType(), new_summands);
      fhe_sub->replaceAllUsesWith(new_sub);
      //TODO: THIS REQUIRES VERIFY-EACH-ZERO SO THE TYPE SYSTEM DOESN'T COMPLAIN TOO MUCH!

      auto new_val = *new_sub->result_begin();

      llvm::outs() << indent << "resolved sub to: ";
      new_val.print(llvm::outs());
      llvm::outs() << indent << "\n";

      return new_val;
    } else if (auto fhe_rot = llvm::dyn_cast<fhe::RotateOp>(op)) {
      //already solved

      llvm::outs() << indent << "resolved rotation to itself: ";
      v.print(llvm::outs());
      llvm::outs() << indent << "\n";

      return v;
    } else if (auto fhe_const = llvm::dyn_cast<fhe::ConstOp>(op)) {
      //TODO: What to do about constants?

      llvm::outs() << indent << "resolved constant to itself: ";
      v.print(llvm::outs());
      llvm::outs() << indent << "\n";

      return v;
    } else {
      emitError(op->getLoc(), "UNEXPECTED OPERATION.");
      return v;
    }
  } else {
    //ignore, not defined in this part of code (e.g., might be a function param).

    llvm::outs() << indent << "ignored: ";
    v.print(llvm::outs());
    llvm::outs() << indent << "\n";

    return v;
  }
}

template<typename OpType>
LogicalResult batchArithmeticOperation(IRRewriter &rewriter, MLIRContext *context, OpType op) {
  // We care only about ops that return scalars, assuming others are already "SIMD-compatible"
  if (auto result_st = op.getType().template dyn_cast_or_null<fhe::SecretType>()) {

    llvm::outs() << "updating ";
    op.print(llvm::outs());
    llvm::outs() << "\n";

    // Move rewriter
    rewriter.setInsertionPointAfter(op);

    /// Target Slot (-1 => no target slot required)
    int target_slot = -1;

    // convert all operands to batched
    for (auto it = op->operand_begin(); it!=op.operand_end(); ++it) {

      if (auto bst = (*it).getType().template dyn_cast_or_null<fhe::BatchedSecretType>()) {
        // already vector-style input, assume it's already slot-aligned appropriately
      } else if (auto st = (*it).getType().template dyn_cast_or_null<fhe::SecretType>()) {
        // scalar-type input that needs to be converted
        if (auto ex_op = (*it).template getDefiningOp<fhe::ExtractOp>()) {
          // instead of using the extract op, issue a rotation instead
          auto i = (int) ex_op.i().getLimitedValue();
          if (target_slot==-1) //no other target slot defined yet, let's make this the target
            target_slot = i; // we'll rotate by zero, but that's later canonicalized to no-op anyway
          auto rotate_op = rewriter
              .create<fhe::RotateOp>(ex_op.getLoc(), ex_op.vector(), i - target_slot);
          rewriter.replaceOpWithIf(ex_op, {rotate_op}, [&](OpOperand &operand) { return operand.getOwner()==op; });
        } else if (auto c_op = (*it).template getDefiningOp<fhe::ConstOp>()) {
          emitError(op.getLoc(),
                    "Encountered not yet supported constant op as defining op for secret operand while trying to batch.");
          return failure();
        } else {
          emitError(op.getLoc(), "Encountered unexpected defining op for secret operand while trying to batch.");
          return failure();
        }

      } else {
        emitError(op.getLoc(), "Encountered unexpected non-secret operand while trying to batch.");
        return failure();
      }
    }

    // new op with batched result type
    auto bst = fhe::BatchedSecretType::get(context, result_st.getPlaintextType());
    auto new_op = rewriter.create<OpType>(op.getLoc(), bst, op->getOperands());

    // Now create a scalar again by creating an extract, preserving type constraints
    auto res_ex_op =
        rewriter.create<fhe::ExtractOp>(op.getLoc(),
                                        op.getType(),
                                        new_op.getResult(),
                                        rewriter.getIndexAttr(target_slot));
    op->replaceAllUsesWith(res_ex_op);

    // Finally, remove the original op
    rewriter.eraseOp(op);
  }
  return success();
}

void BatchingPass::runOnOperation() {

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  //Translate Arithmetic Operations
  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    // We must translate in order of appearance for this to work, so we walk manually
    if (f.walk([&](Operation *op) {
      if (auto sub_op = llvm::dyn_cast_or_null<fhe::SubOp>(op))
        if (batchArithmeticOperation<fhe::SubOp>(rewriter, &getContext(), sub_op).failed())
          return WalkResult::interrupt();
      if (auto add_op = llvm::dyn_cast_or_null<fhe::AddOp>(op))
        if (batchArithmeticOperation<fhe::AddOp>(rewriter, &getContext(), add_op).failed())
          return WalkResult::interrupt();
      if (auto mul_op = llvm::dyn_cast_or_null<fhe::MultiplyOp>(op))
        if (batchArithmeticOperation<fhe::MultiplyOp>(rewriter, &getContext(), mul_op).failed())
          return WalkResult::interrupt();
      return WalkResult(success());
    }).wasInterrupted())
      signalPassFailure();
  }

}