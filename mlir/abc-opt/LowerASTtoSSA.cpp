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
  // Read the existing function arguments
  std::vector<mlir::Type> argTypes;
  std::vector<OpOperand> arguments;
  for (auto op: f.parameters().getOps<FunctionParameterOp>()) {
    auto param_type = op.typeAttr().getValue();
    argTypes.push_back(param_type);
  }

  // Create the new builtin.func Op
  rewriter.setInsertionPoint(f);
  auto func_type = rewriter.getFunctionType(argTypes, f.return_typeAttr().getValue());
  auto new_f = rewriter.create<FuncOp>(f.getLoc(), f.name(), func_type);
  new_f.setPrivate();
  auto entryBlock = new_f.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // Enter the arguments into the symbol table
  // This sets curScope in symbolTable to varScope
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);
  for (auto pair: llvm::zip(f.getRegion(0).getOps<FunctionParameterOp>(), entryBlock->getArguments())) {
    auto op = std::get<0>(pair);
    auto arg = std::get<1>(pair);
    auto param_name = op.nameAttr().getValue();
    if (failed(declare(param_name, arg, symbolTable))) {
      mlir::emitError(arg.getLoc(), "Cannot translate FunctionParameter " + param_name + ": name is already taken.");
    }
  }

  // Move ABC Operations over into the new function's entryBlock
  auto abc_block_it = f.body().getOps<abc::BlockOp>();
  if (abc_block_it.begin()==abc_block_it.end() || ++abc_block_it.begin()!=abc_block_it.end()) {
    emitError(f.getLoc(), "Expected exactly one Block inside function!");
  } else {
    auto abc_block = *abc_block_it.begin();
    if (abc_block->getNumRegions()!=1 || !abc_block.body().hasOneBlock()) {
      emitError(abc_block.getLoc(), "ABC BlockOp must contain exactly one region and exactly one Block in that!");
    } else {
      llvm::iplist<Operation> oplist;
      auto &bb = *abc_block.body().getBlocks().begin();
      rewriter.mergeBlocks(&bb, entryBlock);
    }
  }

  // TODO: Go through the block and translate each operation



  // TODO: Remove this after pass translates return properly.
  //  for now create a dummy return
  rewriter.create<mlir::ReturnOp>(f.getLoc(), entryBlock->getArgument(0));

  // Now that we're done, we can remove the original function
  rewriter.eraseOp(f);
}

void LowerASTtoSSAPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();
  target.addLegalOp<mlir::ReturnOp>();
  // target.addIllegalDialect<ABCDialect>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  for (auto f: llvm::make_early_inc_range(block.getOps<FunctionOp>())) {
    convertFunctionOp2FuncOp(f, rewriter, symbolTable);
  }
}