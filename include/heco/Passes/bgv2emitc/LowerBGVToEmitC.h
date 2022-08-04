#ifndef HECO_PASSES_BGV2EMITC_LOWERBGVTOEMITC_H_
#define HECO_PASSES_BGV2EMITC_LOWERBGVTOEMITC_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerBGVToEmitCPass : public mlir::PassWrapper<LowerBGVToEmitCPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "bgv2emitc";
    }
};

#endif // HECO_PASSES_BGV2EMITC_LOWERBGVTOEMITC_H_
