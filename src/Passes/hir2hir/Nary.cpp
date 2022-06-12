#include "heco/Passes/hir2hir/Nary.h"
#include <iostream>
#include <memory>
#include "heco/IR/FHE/FHEDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void NaryPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<
        fhe::FHEDialect, mlir::AffineDialect, func::FuncDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
}

void collapseAdd(fhe::AddOp &op, IRRewriter &rewriter)
{
    for (auto &use : llvm::make_early_inc_range(op.output().getUses()))
    {
        if (auto use_add = llvm::dyn_cast<fhe::AddOp>(*use.getOwner()))
        {
            rewriter.setInsertionPointAfter(use_add.getOperation());
            llvm::SmallVector<Value, 4> new_operands;
            for (auto s : use_add.x())
            {
                if (s != op.output())
                {
                    new_operands.push_back(s);
                }
            }
            for (auto s : op.x())
            {
                new_operands.push_back(s);
            }
            auto new_add = rewriter.create<fhe::AddOp>(use_add->getLoc(), use_add.output().getType(), new_operands);
            use_add.replaceAllUsesWith(new_add.getOperation());
        }
    }
}

void collapseSub(fhe::SubOp &op, IRRewriter &rewriter)
{
    for (auto &use : llvm::make_early_inc_range(op.output().getUses()))
    {
        if (auto use_sub = llvm::dyn_cast<fhe::SubOp>(*use.getOwner()))
        {
            rewriter.setInsertionPointAfter(use_sub.getOperation());
            llvm::SmallVector<Value, 4> new_operands;
            for (auto s : use_sub.x())
            {
                if (s != op.output())
                {
                    new_operands.push_back(s);
                }
            }
            for (auto s : op.x())
            {
                new_operands.push_back(s);
            }
            auto new_sub = rewriter.create<fhe::SubOp>(use_sub->getLoc(), use_sub.output().getType(), new_operands);
            use_sub.replaceAllUsesWith(new_sub.getOperation());
        }
    }
}

void collapseMul(fhe::MultiplyOp &op, IRRewriter &rewriter)
{
    for (auto &use : llvm::make_early_inc_range(op.output().getUses()))
    {
        if (auto use_mul = llvm::dyn_cast<fhe::MultiplyOp>(*use.getOwner()))
        {
            rewriter.setInsertionPointAfter(use_mul.getOperation());
            llvm::SmallVector<Value, 4> new_operands;
            for (auto s : use_mul.x())
            {
                if (s != op.output())
                {
                    new_operands.push_back(s);
                }
            }
            for (auto s : op.x())
            {
                new_operands.push_back(s);
            }
            auto new_mul =
                rewriter.create<fhe::MultiplyOp>(use_mul->getLoc(), use_mul.output().getType(), new_operands);
            use_mul.replaceAllUsesWith(new_mul.getOperation());
        }
    }
}

void NaryPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalOp<AffineForOp>();

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    // TODO: There's very likely a much better way to do this that's not this kind of manual walk!

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<arith::SubIOp>()))
        {
            rewriter.setInsertionPointAfter(op.getOperation());
            llvm::SmallVector<Value, 2> operands = { op.getLhs(), op.getRhs() };
            Value value = rewriter.create<fhe::SubOp>(op.getLoc(), op.getResult().getType(), operands);
            op.replaceAllUsesWith(value);
            rewriter.eraseOp(op.getOperation());
        }
    }

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<arith::MulIOp>()))
        {
            rewriter.setInsertionPointAfter(op.getOperation());
            llvm::SmallVector<Value, 2> operands = { op.getLhs(), op.getRhs() };
            Value value = rewriter.create<fhe::MultiplyOp>(op.getLoc(), op.getResult().getType(), operands);
            op.replaceAllUsesWith(value);
            rewriter.eraseOp(op.getOperation());
        }
    }

    // Now, go through and actually start collapsing the operations!
    for (auto f : llvm::make_early_inc_range(block.getOps<FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<fhe::AddOp>()))
        {
            collapseAdd(op, rewriter);
        }
    }

    for (auto f : llvm::make_early_inc_range(block.getOps<FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<fhe::SubOp>()))
        {
            collapseSub(op, rewriter);
        }
    }

    for (auto f : llvm::make_early_inc_range(block.getOps<FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<fhe::MultiplyOp>()))
        {
            collapseMul(op, rewriter);
        }
    }

    // TODO: Finally, clean up ops without uses (hard to do during above loop because of iter invalidation)
    //  HACK: just canonicalize again for now ;)
}

// TODO: NOTE WE MUST ADD CSE MANUALLY!