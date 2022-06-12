#ifndef HECO_PASSES_BGV2LLVM_LOWERBGVTOLLVM_H_
#define HECO_PASSES_BGV2LLVM_LOWERBGVTOLLVM_H_

#include "mlir/Pass/Pass.h"

struct LowerBGVToLLVMPass : public mlir::PassWrapper<LowerBGVToLLVMPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "bgv2llvm";
    }
};

#endif // HECO_PASSES_BGV2LLVM_LOWERBGVTOLLVM_H_
