#ifndef ABC_IR_PASSES_LOWERASTTOSSA_H_
#define ABC_IR_PASSES_LOWERASTTOSSA_H_

#include "mlir/Pass/Pass.h"

namespace abc {

/// Lowering from the AST-style ABC dialect to SSA representation
struct LowerASTtoSSAPass : public mlir::PassWrapper<LowerASTtoSSAPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "ast2ssa";
  }
};

}

#endif //ABC_IR_PASSES_LOWERASTTOSSA_H_
