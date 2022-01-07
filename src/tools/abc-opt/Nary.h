#ifndef AST_OPTIMIZER_MLIR_ABC_OPT_NARY_H_
#define AST_OPTIMIZER_MLIR_ABC_OPT_NARY_H_

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

#include "ast_opt/IR/ABC/ABCDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace abc {

struct NaryPass : public mlir::PassWrapper<NaryPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<abc::ABCDialect, mlir::AffineDialect, mlir::StandardOpsDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "nary";
  }
};
}


#endif //AST_OPTIMIZER_MLIR_ABC_OPT_NARY_H_
