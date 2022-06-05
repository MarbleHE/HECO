#ifndef ABC_PASSES_SSA2SSA_SCALARBATCHING_H_
#define ABC_PASSES_SSA2SSA_SCALARBATCHING_H_

#include "mlir/Pass/Pass.h"

struct ScalarBatchingPass : public mlir::PassWrapper<ScalarBatchingPass,
                                                     mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "scalar-batching";
  }
};

#endif //ABC_PASSES_SSA2SSA_SCALARBATCHING_H_
