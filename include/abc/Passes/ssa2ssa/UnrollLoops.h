#ifndef AST_OPTIMIZER_MLIR_ABC_OPT_UNROLLLOOPS_H_
#define AST_OPTIMIZER_MLIR_ABC_OPT_UNROLLLOOPS_H_

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

/// Aggressively tries to unroll all affine.for (AffineForOp) loops
struct UnrollLoopsPass : public mlir::PassWrapper<UnrollLoopsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // This never generates new kinds of operations that weren't previously in the program => no dependent dialects

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "unroll-loops";
  }
};

#endif //AST_OPTIMIZER_MLIR_ABC_OPT_UNROLLLOOPS_H_
