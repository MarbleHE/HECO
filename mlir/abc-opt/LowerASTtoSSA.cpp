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

mlir::Value
translateExpression(Operation & op,
                    IRRewriter & rewriter,
                    llvm::ScopedHashTable<StringRef, mlir::Value> & symbolTable) {
  //TODO:  Actually translate expressions
  auto value = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(1), 1));
  return value;
}

Operation &firstOp(Region &region) {
  return *region.getOps().begin();
}

mlir::Block &getBlock(BlockOp &block_op) {
  if (block_op.body().empty()) {
    block_op.body().emplaceBlock();
  }
  return block_op.body().front();
}

mlir::Block &getBlock(Region &region_containing_blockop) {
  if (region_containing_blockop.empty()) {
    emitError(region_containing_blockop.getLoc(),
              "Expected this region to contain an abc.block but it is empty (no MLIR block).");
  } else if (region_containing_blockop.front().empty()) {
    emitError(region_containing_blockop.getLoc(),
              "Expected this region to contain an abc.block but it is empty (no Ops).");
  } else if (auto block_op = llvm::dyn_cast<BlockOp>(region_containing_blockop.front().front())) {

    if (block_op.body().empty()) {
      // This is valid, but a bit unusual
      block_op.body().emplaceBlock();
    }
    return block_op.body().front();
  } else {
    emitError(region_containing_blockop.getLoc(),
              "Expected this region to contain an abc.block but it contained an "
                  + region_containing_blockop.front().front().getName().getStringRef());
  }
  // Fabricate a block out of thin air so we can always continue on
  region_containing_blockop.emplaceBlock();
  return region_containing_blockop.front();
}

void translateStatement(Operation & op,
                        IRRewriter & rewriter,
                        llvm::ScopedHashTable<StringRef, mlir::Value> & symbolTable) {
  rewriter.setInsertionPoint(&op);
  if (auto block_op = llvm::dyn_cast<abc::BlockOp>(op)) {
    //TODO: Support BlockOp
    //emitError(op.getLoc(), "Nested Blocks are not yet supported.");
  } else if (auto return_op = llvm::dyn_cast<abc::ReturnOp>(op)) {
    if (return_op.getNumRegions() > 0) {
      auto &return_value_expr = firstOp(return_op.value().front());
      rewriter.create<mlir::ReturnOp>(op.getLoc(), translateExpression(return_value_expr, rewriter, symbolTable));
    } else {
      rewriter.create<mlir::ReturnOp>(op.getLoc());
    }
    rewriter.eraseOp(&op);
  } else if (auto assignment_op = llvm::dyn_cast<abc::AssignmentOp>(op)) {
    //TODO: Support AssignmentOp
    //emitError(op.getLoc(), "Op not yet supported.");
  } else if (auto vardecl_op = llvm::dyn_cast<abc::VariableDeclarationOp>(op)) {
    //TODO: Support VariableDeclarationOp
    //emitError(op.getLoc(), "Op not yet supported.");
  } else if (auto for_op = llvm::dyn_cast<abc::ForOp>(op)) {
    //TODO: Support ForOp
    //emitError(op.getLoc(), "Op not yet supported.");
  } else if (auto if_op = llvm::dyn_cast<abc::IfOp>(op)) {
    auto condition = translateExpression(firstOp(if_op.condition()), rewriter, symbolTable);
    bool else_branch = if_op->getNumRegions()==3;
    auto new_if = rewriter.create<scf::IfOp>(if_op->getLoc(), condition, else_branch);

    //THEN
    rewriter.mergeBlocks(&getBlock(if_op.thenBranch()), new_if.thenBlock());
    for (auto &inner_op: llvm::make_early_inc_range(new_if.thenBlock()->getOperations())) {
      translateStatement(inner_op, rewriter, symbolTable);
    }
    // TODO: Handle setting values properly!
    rewriter.setInsertionPointToEnd(new_if.thenBlock());
    rewriter.create<scf::YieldOp>(if_op->getLoc());

    // ELSE
    if (else_branch) {
      rewriter.mergeBlocks(&getBlock(if_op.elseBranch().front()), new_if.elseBlock());
      for (auto &inner_op: llvm::make_early_inc_range(new_if.elseBlock()->getOperations())) {
        translateStatement(inner_op, rewriter, symbolTable);
      }
      // TODO: Handle setting values properly!
      rewriter.setInsertionPointToEnd(new_if.elseBlock());
      rewriter.create<scf::YieldOp>(if_op->getLoc());
    }

    rewriter.eraseOp(&op);

  } else if (auto yield_op = llvm::dyn_cast<scf::YieldOp>(op)) {
    // Do nothing
  } else {
    emitError(op.getLoc(), "Unexpected Op encountered: " + op.getName().getStringRef());
  }
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

  // Now we can remove the original function
  rewriter.eraseOp(f);

  // Finally, go through the block and translate each operation
  for (auto &op: llvm::make_early_inc_range(entryBlock->getOperations())) {
    translateStatement(op, rewriter, symbolTable);
  }
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

  getOperation()->dump();
}