#ifndef AST_OPTIMIZER_MLIR_ABC_OPT_NARY_H_
#define AST_OPTIMIZER_MLIR_ABC_OPT_NARY_H_

#include "mlir/Pass/Pass.h"

struct NaryPass : public mlir::PassWrapper<NaryPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "nary";
  }
};



#endif //AST_OPTIMIZER_MLIR_ABC_OPT_NARY_H_
