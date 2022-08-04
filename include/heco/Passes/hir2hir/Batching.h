#ifndef HECO_PASSES_HIR2HIR_BATCHING_H_
#define HECO_PASSES_HIR2HIR_BATCHING_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct BatchingPass : public mlir::PassWrapper<BatchingPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "batching";
    }
};

#endif // HECO_PASSES_HIR2HIR_BATCHING_H_
