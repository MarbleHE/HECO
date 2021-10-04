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

/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
mlir::LogicalResult declare(llvm::StringRef name,
                            mlir::Value value,
                            llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable) {
  if (symbolTable.count(name))
    return mlir::failure();
  symbolTable.insert(name, value);
  return mlir::success();
}

void convertFunctionOp2FuncOp(FunctionOp &f,
                              IRRewriter &rewriter,
                              llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable) {
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
  // This sets curScope in symbolTable to varScope
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);
  for (auto pair: llvm::zip(f.getRegion(0).getOps<FunctionParameterOp>(), entryBlock->getArguments())) {
    auto op = std::get<0>(pair);
    auto arg = std::get<1>(pair);
    auto param_name = op.nameAttr().getValue();
    if (failed(declare(param_name, arg, symbolTable))) {
      mlir::emitError(arg.getLoc(), "Cannot create FunctionParameter " + param_name + " since name is already taken.");
    }
  }

  rewriter.setInsertionPointToStart(entryBlock);
  auto abc_block_it = f.getRegion(1).getOps<abc::BlockOp>();
  if (abc_block_it.begin()==abc_block_it.end() || ++abc_block_it.begin()!=abc_block_it.end()) {
    emitError(f.getLoc(), "Expected exactly one Block inside function!");
  } else {
    auto abc_block = *abc_block_it.begin();
    llvm::iplist<Operation> oplist;
    assert( abc_block.body().hasOneBlock() && "ABC BlockOp must contain exactly one region and exactly one Block in that!");
    auto &bb = *abc_block.body().getBlocks().begin();
    //TODO: Do something better than just dumping the ABC ops into here
    rewriter.mergeBlocks(&bb, entryBlock);
  }

  // TODO: Fix this, for now create a dummy return
  rewriter.create<mlir::ReturnOp>(f.getLoc(), entryBlock->getArgument(0));

  // Now that we're done, we can remove the original function
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