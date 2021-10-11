#ifndef AST_OPTIMIZER_MLIR_ABC_OPT_LOWERASTTOSSA_H_
#define AST_OPTIMIZER_MLIR_ABC_OPT_LOWERASTTOSSA_H_

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

#include "ABC/ABCDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace abc {

/// Lowering from the AST-style ABC dialect to SSA representation
struct LowerASTtoSSAPass : public mlir::PassWrapper<LowerASTtoSSAPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect, mlir::scf::SCFDialect>();
  }
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "ast2ssa";
  }
};
//
//std::unique_ptr<mlir::Pass> createLowerASTtoSSAPass();
}

#endif //AST_OPTIMIZER_MLIR_ABC_OPT_LOWERASTTOSSA_H_
