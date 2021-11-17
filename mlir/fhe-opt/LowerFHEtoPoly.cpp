#include "LowerFHEtoPoly.h"

#include <iostream>
#include <memory>
#include "llvm/ADT/ScopedHashTable.h"

using namespace mlir;
using namespace fhe;

void LowerFHEtoPolyPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  target.addIllegalDialect<FHEDialect>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  // TODO: DO SOMETHING :)
}