#include "UnrollLoops.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace abc;


void UnrollLoopsPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalOp<AffineForOp>();



}