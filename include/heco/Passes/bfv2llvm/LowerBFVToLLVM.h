#ifndef HECO_PASSES_BFV2LLVM_LOWERBFVTOLLVM_H_
#define HECO_PASSES_BFV2LLVM_LOWERBFVTOLLVM_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerBFVToLLVMPass : public mlir::PassWrapper<LowerBFVToLLVMPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "bfv2llvm";
    }
};

#endif // HECO_PASSES_BFV2LLVM_LOWERBFVTOLLVM_H_
