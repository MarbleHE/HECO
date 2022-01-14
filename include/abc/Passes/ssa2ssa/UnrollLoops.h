#ifndef AST_OPTIMIZER_MLIR_ABC_OPT_UNROLLLOOPS_H_
#define AST_OPTIMIZER_MLIR_ABC_OPT_UNROLLLOOPS_H_

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

/// Aggressively tries to unroll all affine.for (AffineForOp) loops
struct UnrollLoopsPass : public mlir::PassWrapper<UnrollLoopsPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  }
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "unroll-loops";
  }
};

#endif //AST_OPTIMIZER_MLIR_ABC_OPT_UNROLLLOOPS_H_
