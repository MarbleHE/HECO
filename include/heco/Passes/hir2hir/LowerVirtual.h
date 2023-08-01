#ifndef HECO_PASSES_HIR2HIR_LOWERVIRTUAL_H_
#define HECO_PASSES_HIR2HIR_LOWERVIRTUAL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerVirtualPass : public mlir::PassWrapper<LowerVirtualPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "lower-virtual";
    }
};

#endif // HECO_PASSES_HIR2HIR_LOWERVIRTUAL_H_
