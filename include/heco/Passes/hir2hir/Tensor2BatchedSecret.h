#ifndef HECO_PASSES_HIR2HIR_TENSOR2BATCHEDSECRET_H_
#define HECO_PASSES_HIR2HIR_TENSOR2BATCHEDSECRET_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct Tensor2BatchedSecretPass
    : public mlir::PassWrapper<Tensor2BatchedSecretPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;
    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "tensor2fhe";
    }
};

#endif // HECO_PASSES_HIR2HIR_TENSOR2BATCHEDSECRET_H_
