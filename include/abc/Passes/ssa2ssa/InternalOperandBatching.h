#ifndef ABC_PASSES_SSA2SSA_INTERNALOPERANDBATCHING_H_
#define ABC_PASSES_SSA2SSA_INTERNALOPERANDBATCHING_H_

#include "mlir/Pass/Pass.h"

struct InternalOperandBatchingPass : public mlir::PassWrapper<InternalOperandBatchingPass,
                                                              mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "internal-batching";
  }
};

#endif //ABC_PASSES_SSA2SSA_INTERNALOPERANDBATCHING_H_
