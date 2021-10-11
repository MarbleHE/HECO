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

mlir::Value
translateExpression(Operation &op,
                    IRRewriter &rewriter,
                    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable) {
  if (auto literal_int = llvm::dyn_cast<abc::LiteralIntOp>(op)) {
    auto value = rewriter
        .create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), literal_int.value()));
    return value;
  } else if (auto variable = llvm::dyn_cast<abc::VariableOp>(op)) {
    return symbolTable.lookup(variable.name());
  } else {
    //TODO:  Actually translate expressions
    emitError(op.getLoc(), "Expression not yet supported.");
    auto value = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(1), 1));
    return value;
  }

}

StringRef translateTarget(Operation &op,
                          IRRewriter &rewriter,
                          llvm::ScopedHashTable<StringRef, Value> &symbolTable) {

  if (auto variable_op = llvm::dyn_cast<abc::VariableOp>(op)) {
    return variable_op.name();
  } else {
    // TODO: Support other targets (especially arrays!)
    emitError(op.getLoc(),
              "Currently, only variables are supported as assignment targets (got: " + op.getName().getStringRef()
                  + ").");
    return "INVALID_TARGET";
  }
}

void translateStatement(Operation &op,
                        IRRewriter &rewriter,
                        llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
                        AffineForOp* for_op = nullptr);

void translateIfOp(abc::IfOp &if_op, IRRewriter &rewriter, llvm::ScopedHashTable<StringRef, Value> &symbolTable) {
  auto condition = translateExpression(firstOp(if_op.condition()), rewriter, symbolTable);
  bool else_branch = if_op->getNumRegions()==3;
  auto new_if = rewriter.create<scf::IfOp>(if_op->getLoc(), condition, else_branch);

  //THEN
  rewriter.mergeBlocks(&getBlock(if_op.thenBranch()), new_if.thenBlock());
  for (auto &inner_op: llvm::make_early_inc_range(new_if.thenBlock()->getOperations())) {
    translateStatement(inner_op, rewriter, symbolTable);
  }
  // TODO: Handle setting values properly!

  // ELSE
  if (else_branch) {
    rewriter.mergeBlocks(&getBlock(if_op.elseBranch().front()), new_if.elseBlock());
    for (auto &inner_op: llvm::make_early_inc_range(new_if.elseBlock()->getOperations())) {
      translateStatement(inner_op, rewriter, symbolTable);
    }
    // TODO: Handle setting values properly!
  }
}

void translateVariableDeclarationOp(abc::VariableDeclarationOp vardecl_op,
                                    IRRewriter &rewriter,
                                    llvm::ScopedHashTable<StringRef, Value> &symbolTable) {

  if (vardecl_op.value().empty()) {
    emitError(vardecl_op.getLoc(), "Declarations that do not specify a value are currently not supported.");
    return;
  }
  // Get Name, Type and Value
  auto name = vardecl_op.name();
  //auto type = vardecl_op.type();
  // TODO: Support decls without value by defining default values?
  auto value = translateExpression(firstOp(vardecl_op.value().front()), rewriter, symbolTable);
  value.setLoc(NameLoc::get(Identifier::get(name, value.getContext()), value.getLoc()));
  // TODO: Somehow check that value and type are compatible
  (void) declare(name, value, symbolTable); //void cast to suppress "unused result" warning
}

void translateAssignmentOp(abc::AssignmentOp assignment_op,
                           IRRewriter &rewriter,
                           llvm::ScopedHashTable<StringRef, Value> &symbolTable,
                           AffineForOp* for_op) {
  // Get Name, Type and Value
  auto target = translateTarget(firstOp(assignment_op.target()), rewriter, symbolTable);
  auto value = translateExpression(firstOp(assignment_op.value()), rewriter, symbolTable);

  if(for_op) {

    //TODO: check if the symbol table still contains the symbol at the parent scope.
    // If yes, then it's not loop local and we need to do some yield stuff!
    // Next, we should check if it's already been added to the iter_args!
    // by checking if one of the iter args is the same value as the one we get by looking up the old value
    // Finally, if we ARE updating an existing iter arg, we need to find the existing yield stmt and change it
    // otherwise, we can just emit a new yield at the end of the loop
    // However, this might be BAD in terms of iterator stuff since we're currently in an llvm:: make early inc range thing
    // iterating over all the ops nested in this for op!
    emitError(assignment_op->getLoc(), "Currently, we do not handle writing to variables in for loops correctly");
    symbolTable.insert(target, value);
  } else {
    symbolTable.insert(target, value);
  }

}

void translateSimpleForOp(abc::SimpleForOp &simple_for_op,
                          IRRewriter &rewriter,
                          llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable) {

  // Create a new scope
  // This sets curScope in symbolTable to varScope
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

  // Create the affine for loop
  auto new_for = rewriter.create<AffineForOp>(simple_for_op->getLoc(),
                                              simple_for_op.start().getLimitedValue(),
                                              simple_for_op.end().getLimitedValue());


  declare(simple_for_op.iv(), new_for.getInductionVar(), symbolTable);

  // Move ABC Operations over into the new for loop's entryBlock
  rewriter.setInsertionPointToStart(new_for.getBody());
  auto abc_block_it = simple_for_op.body().getOps<abc::BlockOp>();
  if (abc_block_it.begin()==abc_block_it.end() || ++abc_block_it.begin()!=abc_block_it.end()) {
    emitError(simple_for_op.getLoc(), "Expected exactly one Block inside function!");
  } else {
    auto abc_block = *abc_block_it.begin();
    if (abc_block->getNumRegions()!=1 || !abc_block.body().hasOneBlock()) {
      emitError(abc_block.getLoc(), "ABC BlockOp must contain exactly one region and exactly one Block in that!");
    } else {
      llvm::iplist<Operation> oplist;
      auto &bb = *abc_block.body().getBlocks().begin();
      rewriter.mergeBlockBefore(&bb, &new_for.getBody()->front());
    }
  }

  // Finally, go through the block and translate each operation
  // It's the responsiblity of VariableAssignment to update the iterArgs, so we pass this operation along
  for (auto &op: llvm::make_early_inc_range(new_for.getBody()->getOperations())) {
    translateStatement(op, rewriter, symbolTable, &new_for);
  }
}

//}
//void translateForOp(abc::ForOp &for_op,
//                    IRRewriter &rewriter,
//                    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable) {
//
//  auto condition = translateExpression(firstOp(for_op.condition()), rewriter, symbolTable);
//
//  //TODO: support loops!
//  // For now we assume a loop has pattern for({VariableDecl}, {ExprOp}, {AssignmentOp (to same Variable)})
//
////  // Get lower bound:
////  Value lower_bound;
////  StringRef lower_bound_var_name = "";
////  auto &test_op = *llvm::dyn_cast<abc::BlockOp>(firstOp(for_op.initializer())).getOps().begin();
////  if (auto vardecl_op = llvm::dyn_cast<VariableDeclarationOp>(test_op)) {
////    lower_bound_var_name = vardecl_op.name();
////    lower_bound = translateExpression(firstOp(vardecl_op.value().front()), rewriter, symbolTable);
////  } else {
////    emitError(for_op->getLoc(),
////              "Currently we do not support non-trivial loop initializers. Set lower bound to 0 (got "
////                  + test_op.getName().getStringRef() + ").");
////    // Create a dummy initializer so that things can continue.
////    lower_bound =
////        rewriter.create<ConstantOp>(for_op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
////  }
////  if (++for_op.initializer().getOps().begin()!=for_op.initializer().getOps().end()) {
////    emitError(for_op->getLoc(), "Currently we do not support multiple statements in the initializer!.");
////  }
////
////  auto new_for = rewriter.create<scf::ForOp>(for_op->getLoc(),lower_bound, lower_bound);
//
//}

void translateStatement(Operation &op,
                        IRRewriter &rewriter,
                        llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
                        AffineForOp* for_op) {
  rewriter.setInsertionPoint(&op);
  if (auto block_op = llvm::dyn_cast<abc::BlockOp>(op)) {
    //TODO: Support BlockOp
    emitError(op.getLoc(), "Nested Blocks are not yet supported.");
  } else if (auto return_op = llvm::dyn_cast<abc::ReturnOp>(op)) {
    if (return_op.getNumRegions() > 0) {
      auto &return_value_expr = firstOp(return_op.value().front());
      rewriter.create<mlir::ReturnOp>(op.getLoc(), translateExpression(return_value_expr, rewriter, symbolTable));
    } else {
      rewriter.create<mlir::ReturnOp>(op.getLoc());
    }
    rewriter.eraseOp(&op);
  } else if (auto assignment_op = llvm::dyn_cast<abc::AssignmentOp>(op)) {
    translateAssignmentOp(assignment_op, rewriter, symbolTable, for_op);
    rewriter.eraseOp(&op);
  } else if (auto vardecl_op = llvm::dyn_cast<abc::VariableDeclarationOp>(op)) {
    translateVariableDeclarationOp(vardecl_op, rewriter, symbolTable);
    rewriter.eraseOp(&op);
  } else if (auto for_op = llvm::dyn_cast<abc::ForOp>(op)) {
    //TODO: Support general ForOp
    emitError(op.getLoc(), "General For Statements are not yet supported.");
  } else if (auto if_op = llvm::dyn_cast<abc::IfOp>(op)) {
    translateIfOp(if_op, rewriter, symbolTable);
    rewriter.eraseOp(&op);
  } else if (auto scf_yield_op = llvm::dyn_cast<scf::YieldOp>(op)) {
    // Do nothing
  } else if (auto affine_yield_op = llvm::dyn_cast<AffineYieldOp>(op)) {
    // do nothing
  } else if (auto simple_for_op = llvm::dyn_cast<abc::SimpleForOp>(op)) {
    translateSimpleForOp(simple_for_op, rewriter, symbolTable);
    rewriter.eraseOp(&op);
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
}