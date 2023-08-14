#ifndef HECO_PASSES_BFV2EMITC_LOWERBFVTOEMITC_H_
#define HECO_PASSES_BFV2EMITC_LOWERBFVTOEMITC_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerBFVToEmitCPass : public mlir::PassWrapper<LowerBFVToEmitCPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "bfv2emitc";
    }
};

#endif // HECO_PASSES_BFV2EMITC_LOWERBFVTOEMITC_H_
