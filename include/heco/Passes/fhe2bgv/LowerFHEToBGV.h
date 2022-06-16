#ifndef HECO_PASSES_FHE2BGV_LOWERFHETOBGV_H_
#define HECO_PASSES_FHE2BGV_LOWERFHETOBGV_H_

#include "mlir/Pass/Pass.h"

struct LowerFHEToBGVPass : public mlir::PassWrapper<LowerFHEToBGVPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "fhe2bgv";
    }
};

#endif // HECO_PASSES_FHE2BGV_LOWERFHETOBGV_H_
