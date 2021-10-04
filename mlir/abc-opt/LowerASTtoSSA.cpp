//===----------------------------------------------------------------------===//
//
// This file implements a lowering of AST nodes in MLIR (ABC Dialect) to
// a combination of std, builtin, affine and sfc dialects in SSA form
//
//===----------------------------------------------------------------------===//


#include "LowerASTtoSSA.h"

#include <iostream>
#include "llvm/ADT/ScopedHashTable.h"

using namespace mlir;
using namespace abc;

void convertFunctionOp2FuncOp(FunctionOp &f,
                              IRRewriter &rewriter,
                              llvm::ScopedHashTable<StringRef, mlir::Value>& symbolTable) {
  std::vector<mlir::Type> argTypes;
  std::vector<OpOperand> arguments;
  for (auto op: f.getRegion(0).getOps<FunctionParameterOp>()) {
    auto param_type = op.typeAttr().getValue();
    argTypes.push_back(param_type);
  }
  rewriter.setInsertionPoint(f);
  auto func_type = rewriter.getFunctionType(argTypes, f.return_typeAttr().getValue());
  auto new_f = rewriter.create<FuncOp>(f.getLoc(), f.name(), func_type);
  new_f.setPrivate();

  auto entryBlock = new_f.addEntryBlock();
  for (auto pair: llvm::zip(f.getRegion(0).getOps<FunctionParameterOp>(), entryBlock->getArguments())) {
    //TODO: Add these to the converter's symbol table
    auto op = std::get<0>(pair);
    auto arg = std::get<1>(pair);
    auto param_name = op.nameAttr().getValue();
  }

  //TODO: fill the entry block by going through all operations
  rewriter.setInsertionPointToStart(entryBlock);
  rewriter.create<mlir::ReturnOp>(f.getLoc(), entryBlock->getArgument(0));

  // Now that we're done, we can remove the orginal function
  rewriter.eraseOp(f);
}

void LowerASTtoSSAPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  target.addLegalOp<mlir::ReturnOp>();
  // target.addIllegalDialect<ABCDialect>();

  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  for (auto f: llvm::make_early_inc_range(block.getOps<FunctionOp>())) {
    convertFunctionOp2FuncOp(f, rewriter, symbolTable);
  }

  // TODO: Lower the bodies of the FuncOPs, which are still ABC/AST

  // Next approach: Manually walking the IR

//  // Now that the conversion target has been defined, we just need to provide
//  // the set of patterns that will lower the Toy operations.
//  RewritePatternSet patterns(&getContext());
//  patterns.add<ReturnOpLowering>(&getContext());
//  patterns.add<FunctionOpLowering>(&getContext());

//  // With the target and rewrite patterns defined, we can now attempt the
//  // conversion. The conversion will signal failure if any of our `illegal`
//  // operations were not converted successfully.
//  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
//    signalPassFailure();
}