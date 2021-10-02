//===- abc-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ABC/ABCDialect.h"

using namespace mlir;
using namespace abc;

//#include "mlir/Transforms/DialectConversion.h"
//struct AstToSSALoweringPass : public ConversionTarget {
//  AstToSSALoweringPass(MLIRContext &ctx) : ConversionTarget(ctx) {
//    //--------------------------------------------------------------------------
//    // Marking an operation as Legal:
//
//    /// Mark all operations within the LLVM dialect are legal.
//    addLegalDialect<LLVMDialect>();
//
//    /// Mark `std.constant` op is always legal on this target.
//    addLegalOp<ConstantOp>();
//
//    //--------------------------------------------------------------------------
//    // Marking an operation as dynamically legal.
//
//    /// Mark all operations within Affine dialect have dynamic legality
//    /// constraints.
//    addDynamicallyLegalDialect<AffineDialect>([](Operation *op) { ... });
//
//    /// Mark `std.return` as dynamically legal, but provide a specific legality
//    /// callback.
//    addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) { ... });
//
//    /// Treat unknown operations, i.e. those without a legalization action
//    /// directly set, as dynamically legal.
//    markUnknownOpDynamicallyLegal([](Operation *op) { ... });
//
//    //--------------------------------------------------------------------------
//    // Marking an operation as illegal.
//
//    /// All operations within the GPU dialect are illegal.
//    addIllegalDialect<GPUDialect>();
//
//    /// Mark `std.br` and `std.cond_br` as illegal.
//    addIllegalOp<BranchOp, CondBranchOp>();
//  }
//
//  /// Implement the default legalization handler to handle operations marked as
//  /// dynamically legal that were not provided with an explicit handler.
//  bool isDynamicallyLegal(Operation *op) override { ... }
//};
//
//void AstToSSALoweringPass::runOnFunction() {
//  // The first thing to define is the conversion target. This will define the
//  // final target for this lowering.
//  mlir::ConversionTarget target(getContext());
//
//  // We define the specific operations, or dialects, that are legal targets for
//  // this lowering. In our case, we are lowering to a combination of the
//  // `Affine`, `MemRef` and `Standard` dialects.
//  target.addLegalDialect<mlir::AffineDialect, mlir::memref::MemRefDialect,
//                         mlir::StandardOpsDialect>();
//
//  // We also define the Toy dialect as Illegal so that the conversion will fail
//  // if any of these operations are *not* converted. Given that we actually want
//  // a partial lowering, we explicitly mark the Toy operations that don't want
//  // to lower, `toy.print`, as *legal*.
//  target.addIllegalDialect<ToyDialect>();
//  target.addLegalOp<PrintOp>();
//  ...
//}

int main(int argc, char **argv) {
  registerAllPasses();
  // TODO: Register abc optimization passes here.

  mlir::DialectRegistry registry;
  registry.insert<ABCDialect>();
  registry.insert<StandardOpsDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "ABC optimizer driver\n", registry));
}
