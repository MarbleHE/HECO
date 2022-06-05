#ifndef ABC_PASSES_SSA2SSA_BATCHING_H_
#define ABC_PASSES_SSA2SSA_BATCHING_H_

#include "mlir/Pass/Pass.h"

struct BatchingPass : public mlir::PassWrapper<BatchingPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "batching";
  }
};

#endif //ABC_PASSES_SSA2SSA_BATCHING_H_
