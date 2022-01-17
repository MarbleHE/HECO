#ifndef ABC_PASSES_SSA2CPP_LOWERTOEMITC_H_
#define ABC_PASSES_SSA2CPP_LOWERTOEMITC_H_

#include "mlir/Pass/Pass.h"

struct LowerToEmitCPass : public mlir::PassWrapper<LowerToEmitCPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "fhe2cpp";
  }
};

#endif //ABC_PASSES_SSA2CPP_LOWERTOEMITC_H_
