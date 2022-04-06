#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "abc/IR/FHE/FHEDialect.h"
#include "abc/Passes/ssa2ssa/ScalarBatching.h"

using namespace mlir;

void ScalarBatchingPass::getDependentDialects(mlir::DialectRegistry &registry) const {
  registry.insert<fhe::FHEDialect,
                  mlir::AffineDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}


void ScalarBatchingPass::runOnOperation() {

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  for (auto f: llvm::make_early_inc_range(block.getOps<FuncOp>())) {
    // We must translate in order of appearance for this to work, so we walk manually
    if (f.walk([&](Operation *op) {
      return WalkResult(success());
    }).wasInterrupted())
      signalPassFailure();
  }

}