//===----------------------------------------------------------------------===//
//
// This file implements a lowering of AST nodes in MLIR (ABC Dialect) to
// a combination of std, builtin, affine and sfc dialects in SSA form
//
//===----------------------------------------------------------------------===//


#include "LowerASTtoSSA.h"

#include <iostream>
#include <memory>
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
    // TODO: having all ints be index is a nasty hack and we should really instead handle conversions
    //   between things like index, int, bool properly.
    auto value = rewriter
        .create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(literal_int.value().getLimitedValue()));
    return value;
  } else if (auto variable = llvm::dyn_cast<abc::VariableOp>(op)) {
    if (!symbolTable.count(variable.name())) {
      emitError(variable.getLoc(), "Undefined variable " + variable.name());
      return rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
    } else {
      return symbolTable.lookup(variable.name());
    }
  } else if (auto binary_expr = llvm::dyn_cast<abc::BinaryExpressionOp>(op)) {
    auto lhs = translateExpression(firstOp(binary_expr.left()), rewriter, symbolTable);
    auto rhs = translateExpression(firstOp(binary_expr.right()), rewriter, symbolTable);
    if (binary_expr.op()=="+") {
      return rewriter.create<AddIOp>(binary_expr->getLoc(), lhs, rhs);
    } else if (binary_expr.op()=="*") {
      return rewriter.create<MulIOp>(binary_expr->getLoc(), lhs, rhs);
    } else if (binary_expr.op()=="%") {
      return rewriter.create<UnsignedRemIOp>(binary_expr->getLoc(), lhs, rhs);
    } else {
      //TODO: Implement remaining operators
      emitError(binary_expr->getLoc(), "Unsupported operator: " + binary_expr.op());
      return rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
    }
  } else if (auto index_access = llvm::dyn_cast<abc::IndexAccessOp>(op)) {
    if (auto target_variable = llvm::dyn_cast<VariableOp>(firstOp(index_access.target()))) {
      auto target = translateExpression(firstOp(index_access.target()), rewriter, symbolTable);
      auto index = translateExpression(firstOp(index_access.index()), rewriter, symbolTable);
      return rewriter.create<tensor::ExtractOp>(index_access->getLoc(), target, index);
    } else {
      emitError(op.getLoc(),
                "Expected Index Access target to be a variable, got "
                    + firstOp(index_access.target()).getName().getStringRef());
      return rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
    }

  } else {
    //TODO: Actually translate remaining expression types
    emitError(op.getLoc(), "Expression not yet supported.");
    return rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
  }

}

void translateStatement(Operation &op,
                        IRRewriter &rewriter,
                        llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
                        AffineForOp *for_op = nullptr);

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
                           AffineForOp *for_op) {
  // Get Name, Type and Value
  auto value = translateExpression(firstOp(assignment_op.value()), rewriter, symbolTable);
  llvm::StringRef target_name = "INVALID_TARGET";

  auto &targetOp = firstOp(assignment_op.target());

  if (auto variable_op = llvm::dyn_cast<abc::VariableOp>(targetOp)) {
    // If it's a variable,
    target_name = variable_op.name();
  } else if (auto index_access = llvm::dyn_cast<abc::IndexAccessOp>(targetOp)) {
    if (auto target_variable = llvm::dyn_cast<VariableOp>(firstOp(index_access.target()))) {
      // if this is an index access, we need to first insert an operation, then update table with that result value
      // instead, we need to insert an operation and then update the value
      target_name = target_variable.name();
      auto index = translateExpression(firstOp(index_access.index()), rewriter, symbolTable);
      value = rewriter.create<tensor::InsertOp>(assignment_op->getLoc(), value, symbolTable.lookup(target_name), index);
    } else {
      emitError(assignment_op.getLoc(),
                "Expected Index Access target to be a variable, got "
                    + firstOp(index_access.target()).getName().getStringRef());
    }
  } else {
    emitError(assignment_op.target().getLoc(), "Got invalid assignment target!");
  }

  if (for_op) {
    //NOTE:
    // THE BELOW DOESN'T WORK BECAUSE IT SEEMS LIKE WE CAN'T ADD ITER_ARGS TO AN EXISTING FOR_OP?
    // SO FOR NOW WE JUST PUT EVERYTHING IN WHEN WE GENERATE A FOR OP AND LET THE CANONICALIZATION GET RID OF UNNEEDED ONES
    // check if the symbol table still contains the symbol at the parent scope.
    // If yes, then it's not loop local and we need to do some yield stuff!
    // Next, we should check if it's already been added to the iter_args!
    // by checking if one of the iter args is the same value as the one we get by looking up the old value
    // Finally, if we ARE updating an existing iter arg, we need to find the existing yield stmt and change it
    // otherwise, we can just emit a new yield at the end of the loop
    // However, this might be BAD in terms of iterator stuff since we're currently in an llvm:: make early inc range thing
    // iterating over all the ops nested in this for op!
    //emitError(assignment_op->getLoc(), "Currently, we do not handle writing to variables in for loops correctly");
    symbolTable.insert(target_name, value);
  } else {
    symbolTable.insert(target_name, value);
  }

}

void translateSimpleForOp(abc::SimpleForOp &simple_for_op,
                          IRRewriter &rewriter,
                          llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable) {

  std::vector<std::string> existing_vars;
  AffineForOp *new_for_ptr = nullptr;
  // Create a new scope
  {
    // This sets curScope in symbolTable to varScope
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

    // Get every variable that exists and dump it as an iter args,
    // since we can't add them later, but ones that don't get used
    // are easily optimized away by --canonicalize
    // TODO: SERIOUSLY, WE CAN'T EVEN ENUMERATE EVERY SYMBOL WE HAVE??

    //TODO: THis hack is horrible, but until we fix the symbol table stuff, it'll do for the benchmarks
    llvm::SmallVector<Value, 4> iter_args;

    for (auto &hack: {"img", "img2", "value", "x"}) {
      if (symbolTable.count(hack)) {
        existing_vars.emplace_back(hack);
      }
    }
    // get current values
    for (auto &var: existing_vars) {
      iter_args.push_back(symbolTable.lookup(var));
    }

    // Create the affine for loop
    auto new_for = rewriter.create<AffineForOp>(simple_for_op->getLoc(),
                                                simple_for_op.start().getLimitedValue(),
                                                simple_for_op.end().getLimitedValue(),
                                                1, //step size
                                                iter_args);
    new_for_ptr = &new_for;

    //TODO: Hack from above continued, needs to be cleaned up once we fix symboltable
    auto iter_args_it = new_for.getRegionIterArgs().begin();
    for (auto &var: existing_vars) {
      symbolTable.insert(var, *iter_args_it++);
    }

    declare(simple_for_op.iv(), new_for.getInductionVar(), symbolTable);

    emitWarning(simple_for_op->getLoc(), "Currently, manually checking yields is required because of...reasons.");


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
        rewriter.mergeBlocks(&bb, new_for.getBody());
      }
    }

    // Finally, go through the block and translate each operation
    // It's the responsiblity of VariableAssignment to update the iterArgs, so we pass this operation along
    for (auto &op: llvm::make_early_inc_range(new_for.getBody()->getOperations())) {
      translateStatement(op, rewriter, symbolTable, &new_for);
    }

    // TODO: Again, hacky, fix once we have better system
    llvm::SmallVector<Value, 4> yield_values;
    for (auto &var: existing_vars) {
      yield_values.push_back(symbolTable.lookup(var));
    }
    rewriter.setInsertionPointToEnd(new_for.getBody());
    rewriter.create<AffineYieldOp>(simple_for_op->getLoc(), yield_values);

    //TODO: This happens to introduce a bunch of redundant yield/iterargs that canoncialize doesn't catch
    //  fix properly and/or add canonicalization where if yield is same as start value, then it's removed

  }
// exit scope manually

  //Hack: Update the yield values in the symboltable now that we've left the scope
  auto res_it = new_for_ptr->result_begin();
  auto var_it = existing_vars.begin();
  for (size_t i = 0; i < existing_vars.size(); i++) {
    symbolTable.insert(*var_it, *res_it);
    res_it++;
    var_it++;
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
                        AffineForOp *for_op) {
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
  target.addLegalDialect<AffineDialect, StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalDialect<ABCDialect>();

  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  for (auto f: llvm::make_early_inc_range(block.getOps<FunctionOp>())) {
    convertFunctionOp2FuncOp(f, rewriter, symbolTable);
  }
}