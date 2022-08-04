#ifndef HECO_PASSES_HIR2HIR_NARY_H_
#define HECO_PASSES_HIR2HIR_NARY_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct NaryPass : public mlir::PassWrapper<NaryPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;
    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "nary";
    }
};

#endif // HECO_PASSES_HIR2HIR_NARY_H_
