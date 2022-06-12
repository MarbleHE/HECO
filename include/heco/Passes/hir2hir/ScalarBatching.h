#ifndef HECO_PASSES_HIR2HIR_SCALARBATCHING_H_
#define HECO_PASSES_HIR2HIR_SCALARBATCHING_H_

#include "mlir/Pass/Pass.h"

struct ScalarBatchingPass : public mlir::PassWrapper<ScalarBatchingPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "scalar-batching";
    }
};

#endif // HECO_PASSES_HIR2HIR_SCALARBATCHING_H_
