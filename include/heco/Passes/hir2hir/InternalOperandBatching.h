#ifndef HECO_PASSES_HIR2HIR_INTERNALOPERANDBATCHING_H_
#define HECO_PASSES_HIR2HIR_INTERNALOPERANDBATCHING_H_

#include "mlir/Pass/Pass.h"

struct InternalOperandBatchingPass
    : public mlir::PassWrapper<InternalOperandBatchingPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "internal-batching";
    }
};

#endif // HECO_PASSES_HIR2HIR_INTERNALOPERANDBATCHING_H_
