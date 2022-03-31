#ifndef ABC_PASSES_SSA2SSA_COMBINESIMPLIFY_H_
#define ABC_PASSES_SSA2SSA_COMBINESIMPLIFY_H_

#include "mlir/Pass/Pass.h"

struct CombineSimplifyPass : public mlir::PassWrapper<CombineSimplifyPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "combine-simplify";
  }
};

#endif //ABC_PASSES_SSA2SSA_COMBINESIMPLIFY_H_
