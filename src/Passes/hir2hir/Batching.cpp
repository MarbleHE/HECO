#include "heco/Passes/hir2hir/Batching.h"
#include <queue>
#include <unordered_map>
#include "heco/IR/FHE/FHEDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void BatchingPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<
        fhe::FHEDialect, mlir::AffineDialect, func::FuncDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
        mlir::linalg::LinalgDialect>();
}

typedef Value VectorValue;
typedef llvm::SmallMapVector<VectorValue, Value, 1> scalarBatchingMap;

template <typename OpType>
LogicalResult batchArithmeticOperation(IRRewriter &rewriter, MLIRContext *context, OpType op, scalarBatchingMap map)
{
    // We care only about ops that return scalars, assuming others are already "SIMD-compatible"
    if (auto result_st = op.getType().template dyn_cast_or_null<fhe::SecretType>())
    {
        // llvm::outs() << "updating:\n";
        // for (auto o : op.getOperands())
        // {
        //     llvm::outs() << "\t";
        //     o.print(llvm::outs());
        //     llvm::outs() << "\n";
        // }
        // llvm::outs() << "\t";
        // op.print(llvm::outs());
        // llvm::outs() << "\n";

        /// Target Slot (-1 => no target slot required)
        int target_slot = -1;

        // TODO: the code below does the "wrong" thing for hamming distance, because it isn't aware that the add
        //  at the end doesn't actually have any slot preferences. We might be able to fix that during internal-batching
        //  but that sounds like a lot of work/materializing/inserting things.
        //  This issue might be somehow related to relative vs absolute offsets?
        // Find a target slot, if it exists, by iterating through users until we wind either extract, insert or scalar
        // return std::queue<Operation *> users; for (auto u: op->getUsers()) {
        //  users.push(u);
        //}
        // while (!users.empty()) {
        //  auto u = users.front();
        //  users.pop();
        //  if (auto ex_op = dyn_cast_or_null<fhe::ExtractOp>(u)) {
        //    // We later want this value to be in slot i??
        //    target_slot = ex_op.i().getLimitedValue(INT32_MAX);
        //    break;
        //  } else if (auto ins_op = dyn_cast_or_null<fhe::InsertOp>(u)) {
        //    target_slot = ins_op.i().getLimitedValue(INT32_MAX);
        //    break;
        //  } else if (auto ret_op = dyn_cast_or_null<func::ReturnOp>(u)) {
        //    if (ret_op->getOperandTypes().front().template isa<fhe::SecretType>()) {
        //      // we eventually want this as a scalar, which means slot 0
        //      target_slot = 0;
        //      break;
        //    }
        //  }
        //  for (auto uu: u->getUsers()) {
        //    users.push(uu);
        //  }
        //}

        // TODO: as a basic fix, we only search one level deep for now.
        //  This prevents the issue mentioned above but of course might prevent optimizations for complex code
        for (auto u : op->getUsers())
        {
            if (auto ex_op = dyn_cast_or_null<fhe::ExtractOp>(u))
            {
                // We later want this value to be in slot i??
                target_slot = ex_op.i().getLimitedValue(INT32_MAX);
                break;
            }
            else if (auto ins_op = dyn_cast_or_null<fhe::InsertOp>(u))
            {
                target_slot = ins_op.i().getLimitedValue(INT32_MAX);
                break;
            }
            else if (auto ret_op = dyn_cast_or_null<func::ReturnOp>(u))
            {
                if (ret_op->getOperandTypes().front().template isa<fhe::SecretType>())
                    // we eventually want this as a scalar, which means slot 0
                    target_slot = 0;
                break;
            }
        }

        // instead of just picking the first target slot we see, we check if we can find 0
        if (target_slot == -1)
        {
            for (auto it = op->operand_begin(); it != op.operand_end(); ++it)
            {
                if (auto st = (*it).getType().template dyn_cast_or_null<fhe::SecretType>())
                {
                    // scalar-type input that needs to be converted
                    if (auto ex_op = (*it).template getDefiningOp<fhe::ExtractOp>())
                    {
                        auto i = (int)ex_op.i().getLimitedValue();
                        if (target_slot == -1)
                            target_slot = i;
                    }
                }
            }
            // llvm::outs() << "OPERAND-based target slot " << target_slot << "\n";
        }
        else
        {
            // llvm::outs() << "use-based target slot " << target_slot << " \n";
        }
        if (target_slot == -1)
        {
            // llvm::outs() << "NO target slot (" << target_slot << ") \n:";
        }

        // new op
        rewriter.setInsertionPointAfter(op);
        auto new_op = rewriter.create<OpType>(op.getLoc(), op->getOperands());
        rewriter.setInsertionPoint(new_op); // otherwise, any operand transforming ops will be AFTER the new op

        // find the maximum size of vector involved
        int max_size = -1;
        for (auto o : new_op.getOperands())
        {
            if (auto bst = o.getType().template dyn_cast_or_null<fhe::BatchedSecretType>())
            {
                max_size = std::max(max_size, bst.getSize());
            }
            else if (auto st = o.getType().template dyn_cast_or_null<fhe::SecretType>())
            {
                // scalar-type input that will be converted
                if (fhe::ExtractOp ex_op = o.template getDefiningOp<fhe::ExtractOp>())
                {
                    auto bst = ex_op.vector().getType().dyn_cast_or_null<fhe::BatchedSecretType>();
                    assert(bst && "fhe.extract must be applied to BatchedSecret");
                    max_size = std::max(max_size, bst.getSize());
                }
            }
        }

        // convert all operands from scalar to batched
        for (auto it = new_op->operand_begin(); it != new_op.operand_end(); ++it)
        {
            if (auto bst = (*it).getType().template dyn_cast_or_null<fhe::BatchedSecretType>())
            {
                // Check if it needs to be resized
                if (bst.getSize() < max_size)
                {
                    auto resized_type =
                        fhe::BatchedSecretType::get(rewriter.getContext(), bst.getPlaintextType(), max_size);
                    auto resized_o = rewriter.create<fhe::MaterializeOp>(op.getLoc(), resized_type, *it);
                    rewriter.replaceOpWithIf((*it).getDefiningOp(), { resized_o }, [&](OpOperand &operand) {
                        return operand.getOwner() == new_op;
                    });
                }
            }
            else if (auto st = (*it).getType().template dyn_cast_or_null<fhe::SecretType>())
            {
                // scalar-type input that needs to be converted
                if (fhe::ExtractOp ex_op = (*it).template getDefiningOp<fhe::ExtractOp>())
                {
                    // Check if it needs to be resized
                    if (auto bst = ex_op.vector().getType().template dyn_cast_or_null<fhe::BatchedSecretType>())
                    {
                        if (bst.getSize() < max_size)
                        {
                            // llvm::outs() << "\tFound mismatching size.\n\t";
                            // ex_op.print(llvm::outs());
                            // llvm::outs() << "\n\tWill be resized to " << max_size << "\n";
                            auto resized_type =
                                fhe::BatchedSecretType::get(rewriter.getContext(), bst.getPlaintextType(), max_size);
                            auto cur_ip = rewriter.getInsertionPoint();
                            rewriter.setInsertionPoint(ex_op);
                            auto resized_o =
                                rewriter.create<fhe::MaterializeOp>(ex_op.getLoc(), resized_type, ex_op.vector());
                            // llvm::outs() << "\t Created new materialization op:\n\t";
                            // resized_o.print(llvm::outs());
                            // llvm::outs() << "\n";
                            auto resized_ex_op = rewriter.create<fhe::ExtractOp>(
                                ex_op.getLoc(), ex_op.getType(), resized_o, ex_op.iAttr());
                            rewriter.replaceOpWithIf(ex_op, { resized_ex_op }, [&](OpOperand &operand) {
                                return operand.getOwner() == new_op;
                            });
                            ex_op = resized_ex_op;
                            rewriter.setInsertionPoint(&*cur_ip);
                        }
                    }

                    // instead of using the extract op, issue a rotation instead
                    auto i = (int)ex_op.i().getLimitedValue();
                    if (target_slot == -1) // no other target slot defined yet, let's make this the target
                        target_slot = i; // we'll rotate by zero, but that's later canonicalized to no-op anyway
                    auto rotate_op = rewriter.create<fhe::RotateOp>(ex_op.getLoc(), ex_op.vector(), target_slot - i);
                    rewriter.replaceOpWithIf(
                        ex_op, { rotate_op }, [&](OpOperand &operand) { return operand.getOwner() == new_op; });
                    // llvm::outs() << "rewritten operand in op: ";
                    // new_op.print(llvm::outs());
                    // llvm::outs() << '\n';
                }
                else if (auto c_op = (*it).template getDefiningOp<fhe::ConstOp>())
                {
                    // Constant Ops don't take part in resolution?
                    ShapedType shapedType = RankedTensorType::get({ 1 }, c_op.value().getType());
                    auto denseAttr = DenseElementsAttr::get(shapedType, ArrayRef<Attribute>(c_op.value()));
                    auto new_cst = rewriter.template create<fhe::ConstOp>(
                        c_op.getLoc(), fhe::BatchedSecretType::get(context, st.getPlaintextType()), denseAttr);
                    rewriter.replaceOpWithIf(
                        c_op, { new_cst }, [&](OpOperand &operand) { return operand.getOwner() == new_op; });
                }
                else
                {
                    emitWarning(
                        new_op.getLoc(), "Encountered unexpected (non batchable) defining op for secret operand "
                                         "while trying to batch.");
                    return failure();
                }
            }
            else
            {
                // non-secret input, which we can always transform as needed -> no action needed now
            }
        }
        // re-create the op to get correct type inference
        // TODO: avoid this by moving op creation from before operands to after
        auto newer_op = rewriter.create<OpType>(new_op.getLoc(), new_op->getOperands());
        rewriter.eraseOp(new_op);
        new_op = newer_op;

        // SCALAR BATCHING: SEE IF WE CAN DO MAGIC :)
        Value scalar = nullptr;
        fhe::RotateOp rotate = nullptr;
        // Check if we have the scalar +/*/- rotate form that we need
        if (new_op.getNumOperands() == 2)
        {
            if (new_op.getOperand(0).getType().isIntOrIndexOrFloat())
            {
                scalar = new_op.getOperand(0);
                rotate = new_op.getOperand(1).template getDefiningOp<fhe::RotateOp>();
            }
            else if (new_op.getOperand(1).getType().isIntOrIndexOrFloat())
            {
                scalar = new_op.getOperand(1);
                rotate = new_op.getOperand(0).template getDefiningOp<fhe::RotateOp>();
            }
        }
        if (scalar && rotate)
        {
            llvm::outs() << "Attempting scalar batching for:\n";
            llvm::outs() << "\trotate: ";
            rotate.print(llvm::outs());
            llvm::outs() << "\n\tscalar: ";
            scalar.print(llvm::outs());
            llvm::outs() << "\n\ttarget_slot = " << target_slot << "\n";

            // look up if we have an entry for rotate.vector already. If not, create an empty vector
            auto x_map = map.find(rotate.x());
            if (x_map == map.end())
            {
                // create a new tensor
                rewriter.setInsertionPointAfter(new_op);
                auto tensor = rewriter.create<linalg::InitTensorOp>(
                    scalar.getLoc(), ValueRange(), ArrayRef<int64_t>(rotate.getType().getSize()), scalar.getType());
                auto index = rewriter.create<arith::ConstantOp>(
                    scalar.getLoc(), rewriter.getIndexAttr(rotate.i() - target_slot), rewriter.getIndexType());
                auto insert = rewriter.create<tensor::InsertOp>(scalar.getLoc(), scalar, tensor, ValueRange({ index }));
                map.insert({ rotate.x(), insert });
            }
            else
            {
                // TODO: THIS WON'T WORK :(
            }
        }
        // Now create a scalar again by creating an extract, preserving type constraints
        rewriter.setInsertionPointAfter(new_op);
        auto res_ex_op = rewriter.create<fhe::ExtractOp>(
            op.getLoc(), op.getType(), new_op.getResult(), rewriter.getIndexAttr(target_slot));
        op->replaceAllUsesWith(res_ex_op);

        // Finally, remove the original op
        rewriter.eraseOp(op);

        // llvm::outs() << "current function: ";
        // new_op->getParentOp()->print(llvm::outs());
        // llvm::outs() << '\n';
    }
    return success();
}

void BatchingPass::runOnOperation()
{
    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    scalarBatchingMap map;

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if (f.walk([&](Operation *op) {
                 // llvm::outs() << "Walking operation ";
                 // op->print(llvm::outs());
                 // llvm::outs() << "\n";
                 // llvm::outs().flush();
                 if (auto sub_op = llvm::dyn_cast_or_null<fhe::SubOp>(op))
                 {
                     if (batchArithmeticOperation<fhe::SubOp>(rewriter, &getContext(), sub_op, map).failed())
                         return WalkResult::interrupt();
                 }
                 else if (auto add_op = llvm::dyn_cast_or_null<fhe::AddOp>(op))
                 {
                     if (batchArithmeticOperation<fhe::AddOp>(rewriter, &getContext(), add_op, map).failed())
                         return WalkResult::interrupt();
                 }
                 else if (auto mul_op = llvm::dyn_cast_or_null<fhe::MultiplyOp>(op))
                 {
                     if (batchArithmeticOperation<fhe::MultiplyOp>(rewriter, &getContext(), mul_op, map).failed())
                         return WalkResult::interrupt();
                 }
                 // TODO: Add support for relinearization!
                 return WalkResult(success());
             }).wasInterrupted())
            signalPassFailure();
    }
}