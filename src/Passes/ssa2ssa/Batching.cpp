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
          ShapedType shapedType = RankedTensorType::get({1}, c_op.value().getType());
          auto denseAttr = DenseElementsAttr::get(shapedType, ArrayRef<Attribute>(c_op.value()));
          auto new_cst = rewriter.template create<fhe::ConstOp>(c_op.getLoc(),
                                                                fhe::BatchedSecretType::get(context,
                                                                                            st.getPlaintextType()),
                                                                denseAttr);
          rewriter.replaceOpWithIf(c_op, {new_cst}, [&](OpOperand &operand) { return operand.getOwner()==op; });
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