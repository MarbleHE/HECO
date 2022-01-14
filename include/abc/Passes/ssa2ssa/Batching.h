#ifndef AST_OPTIMIZER_MLIR_ABC_OPT_BATCHING_H_
#define AST_OPTIMIZER_MLIR_ABC_OPT_BATCHING_H_

#include "mlir/Pass/Pass.h"

/// Lowering from the AST-style ABC dialect to SSA representation
struct BatchingPass : public mlir::PassWrapper<BatchingPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "batching";
  }
};

#endif //AST_OPTIMIZER_MLIR_ABC_OPT_BATCHING_H_
