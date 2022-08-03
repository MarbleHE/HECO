#ifndef HECO_PASSES_HIR2HIR_COMBINESIMPLIFY_H_
#define HECO_PASSES_HIR2HIR_COMBINESIMPLIFY_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct CombineSimplifyPass : public mlir::PassWrapper<CombineSimplifyPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "combine-simplify";
    }
};

#endif // HECO_PASSES_HIR2HIR_COMBINESIMPLIFY_H_
