#ifndef AST_OPTIMIZER_MLIR_FHE_OPT_LOWERFHETOPOLYH_
#define AST_OPTIMIZER_MLIR_FHE_OPT_LOWERFHETOPOLYH_

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

#include "FHE/FHEDialect.h"
namespace fhe {

/// Lowering from the AST-style ABC dialect to SSA representation
struct LowerFHEtoPolyPass : public mlir::PassWrapper<LowerFHEtoPolyPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect>();
  }
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "fhe2poly";
  }
};
//
//std::unique_ptr<mlir::Pass> createLowerASTtoSSAPass();
}

#endif //AST_OPTIMIZER_MLIR_FHE_OPT_LOWERFHETOPOLYH_
