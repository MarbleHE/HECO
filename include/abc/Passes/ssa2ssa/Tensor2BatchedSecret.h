#ifndef ABC_INCLUDE_ABC_PASSES_SSA2SSA_TENSOR2BATCHEDSECRET_H_
#define ABC_INCLUDE_ABC_PASSES_SSA2SSA_TENSOR2BATCHEDSECRET_H_

#include "mlir/Pass/Pass.h"

struct Tensor2BatchedSecretPass : public mlir::PassWrapper<Tensor2BatchedSecretPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "tensor2batchedsecret";
  }
};

#endif //ABC_INCLUDE_ABC_PASSES_SSA2SSA_TENSOR2BATCHEDSECRET_H_
