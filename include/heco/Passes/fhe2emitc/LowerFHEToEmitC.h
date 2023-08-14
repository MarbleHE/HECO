#ifndef HECO_PASSES_FHE2EMITC_LOWERFHETOEMITC_H_
#define HECO_PASSES_FHE2EMITC_LOWERFHETOEMITC_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerFHEToEmitCPass : public mlir::PassWrapper<LowerFHEToEmitCPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "fhe2emitc";
    }
};

#endif // HECO_PASSES_FHE2EMITC_LOWERFHETOEMITC_H_