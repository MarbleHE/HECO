//===----------------------------------------------------------------------===//
//
// This file implements a lowering of AST nodes in MLIR (ABC Dialect) to
// a combination of fhe, std, builtin, affine and sfc dialects in SSA form
//
//===----------------------------------------------------------------------===//

#include "heco/Passes/ast2ssa/LowerASTtoSSA.h"

#include <memory>
#include <unordered_map>
#include <iostream>
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "heco/IR/AST/ASTDialect.h"
#include "heco/IR/FHE/FHEDialect.h"

using namespace mlir;
using namespace ast;

typedef llvm::ScopedHashTable<StringRef, std::pair<mlir::Type, mlir::Value>> CustomSymbolTable;

/// Go from an Int, Float or Index type to the appropriate Attribute type
Attribute type_to_attr(Type t, Attribute v)
{
  assert(t.isIntOrIndexOrFloat());
  if (t.isIntOrIndex())
  {
    return IntegerAttr::get(t, getConstantIntValue(v).getValue());
  }
  if (t.dyn_cast_or_null<FloatType>())
  {
    if (v.getType().isIntOrFloat() && !v.getType().isIntOrIndex())
    {
      return FloatAttr::get(t, v.cast<FloatAttr>().getValue());
    }
    else
    {
      return FloatAttr::get(t, getConstantIntValue(v).getValue());
    }
  }
  else
  {
    assert(false && "should never be reached.");
  }
}

/// Declare a variable in the current scope, return success if the variable wasn't declared yet.
mlir::LogicalResult declare(llvm::StringRef name, mlir::Type type, mlir::Value value, CustomSymbolTable &symbolTable)
{
  if (symbolTable.count(name))
    return mlir::failure();
  symbolTable.insert(name, {type, value});
  return mlir::success();
}

Operation &firstOp(Region &region)
{
  return *region.getOps().begin();
}

mlir::Block &getBlock(ast::BlockOp &block_op)
{
  if (block_op.body().empty())
  {
    block_op.body().emplaceBlock();
  }
  return block_op.body().front();
}

mlir::Block &getBlock(Region &region_containing_blockop)
{
  if (region_containing_blockop.empty())
  {
    emitError(region_containing_blockop.getLoc(),
              "Expected this region to contain an abc.block but it is empty (no MLIR block).");
  }
  else if (region_containing_blockop.front().empty())
  {
    emitError(region_containing_blockop.getLoc(),
              "Expected this region to contain an abc.block but it is empty (no Ops).");
  }
  else if (auto block_op = llvm::dyn_cast<ast::BlockOp>(region_containing_blockop.front().front()))
  {

    if (block_op.body().empty())
    {
      // This is valid, but a bit unusual
      block_op.body().emplaceBlock();
    }
    return block_op.body().front();
  }
  else
  {
    emitError(region_containing_blockop.getLoc(),
              "Expected this region to contain an abc.block but it contained an " + region_containing_blockop.front().front().getName().getStringRef());
  }
  // Fabricate a block out of thin air in case we found nothing
  // This is necessary to avoid "control reach end of non-void function", since emitError isn't a return/throw.
  region_containing_blockop.emplaceBlock();
  return region_containing_blockop.front();
}

mlir::Value
translateExpression(Operation &op,
                    IRRewriter &rewriter,
                    CustomSymbolTable &symbolTable)
{
  if (auto literal_int = llvm::dyn_cast<ast::LiteralIntOp>(op))
  {
    // Literal Ints are created as "Index" to start with (since it's most common) and should later be converted if needed
    auto value = rewriter
                     .create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(literal_int.value().getLimitedValue()));
    return value;
  }
  else if (auto literal_tensor = llvm::dyn_cast<ast::LiteralTensorOp>(op))
  {
    llvm::SmallVector<int64_t, 4> stuff;
    for (auto i : literal_tensor.value().getValues<IntegerAttr>())
    {
      stuff.push_back(i.getInt());
    }
    auto value = rewriter
                     .create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexTensorAttr(stuff));
    return value;
  }
  else if (auto variable = llvm::dyn_cast<ast::VariableOp>(op))
  {
    if (!symbolTable.count(variable.name()))
    {
      emitError(variable.getLoc(), "Undefined variable " + variable.name());
      return rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
    }
    else
    {
      return symbolTable.lookup(variable.name()).second;
    }
  }
  else if (auto binary_expr = llvm::dyn_cast<ast::BinaryExpressionOp>(op))
  {
    auto lhs = translateExpression(firstOp(binary_expr.left()), rewriter, symbolTable);
    auto rhs = translateExpression(firstOp(binary_expr.right()), rewriter, symbolTable);
    if (binary_expr.op() == "+")
    {
      if (lhs.getType().dyn_cast_or_null<fhe::SecretType>() || rhs.getType().dyn_cast_or_null<fhe::SecretType>())
      {
        return rewriter.create<fhe::AddOp>(binary_expr->getLoc(), ValueRange({lhs, rhs}));
      }
      else
      {
        return rewriter.create<arith::AddIOp>(binary_expr->getLoc(), lhs, rhs);
      }
    }
    else if (binary_expr.op() == "-")
    {
      if (lhs.getType().dyn_cast_or_null<fhe::SecretType>() || rhs.getType().dyn_cast_or_null<fhe::SecretType>())
      {
        return rewriter.create<fhe::SubOp>(binary_expr->getLoc(), ValueRange({lhs, rhs}));
      }
      else
      {
        return rewriter.create<arith::SubIOp>(binary_expr->getLoc(), lhs, rhs);
      }
    }
    else if (binary_expr.op() == "*")
    {
      if (lhs.getType().dyn_cast_or_null<fhe::SecretType>() || rhs.getType().dyn_cast_or_null<fhe::SecretType>())
      {
        return rewriter.create<fhe::MultiplyOp>(binary_expr->getLoc(), ValueRange({lhs, rhs}));
      }
      else
      {
        return rewriter.create<arith::MulIOp>(binary_expr->getLoc(), lhs, rhs);
      }
    }
    else if (binary_expr.op() == "%")
    {
      return rewriter.create<arith::RemUIOp>(binary_expr->getLoc(), lhs, rhs);
    }
    else
    {
      // TODO: Implement remaining operators
      emitError(binary_expr->getLoc(), "Unsupported operator: " + binary_expr.op());
      return rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
    }
  }
  else if (auto index_access = llvm::dyn_cast<ast::IndexAccessOp>(op))
  {
    if (auto target_variable = llvm::dyn_cast<ast::VariableOp>(firstOp(index_access.target())))
    {
      auto target = translateExpression(firstOp(index_access.target()), rewriter, symbolTable);
      auto index = translateExpression(firstOp(index_access.index()), rewriter, symbolTable);
      return rewriter.create<tensor::ExtractOp>(index_access->getLoc(), target, index);
    }
    else if (auto target_ia = llvm::dyn_cast<ast::IndexAccessOp>(firstOp(index_access.target())))
    {
      if (auto nested_target_variable = llvm::dyn_cast<ast::VariableOp>(firstOp(target_ia.target())))
      {
        auto outer_index = translateExpression(firstOp(index_access.index()), rewriter, symbolTable);
        auto inner_index = translateExpression(firstOp(target_ia.index()), rewriter, symbolTable);
        auto inner_target = translateExpression(firstOp(target_ia.target()), rewriter, symbolTable);
        ValueRange indices = {outer_index, inner_index};
        // TODO: WHY NOT NESTED PROPERLY?
        return rewriter.create<tensor::ExtractOp>(index_access->getLoc(), inner_target, inner_index);
      }
      else
      {
        emitError(op.getLoc(),
                  "Expected Index Access target to be nested once or a  variable, got " + firstOp(index_access.target()).getName().getStringRef());
        return {};
      }
    }
    else
    {

      return rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
    }
  }
  else
  {
    // TODO: Translate remaining expression types
    emitError(op.getLoc(), "Expression not yet supported.");
    return {};
  }
}

void translateStatement(Operation &op,
                        IRRewriter &rewriter,
                        CustomSymbolTable &symbolTable,
                        AffineForOp *current_for_op = nullptr,
                        std::function<void(const std::string &, Value, Value)> *if_op_callback = nullptr);

void translateIfOp(ast::IfOp &if_op, IRRewriter &rewriter, CustomSymbolTable &symbolTable)
{
  auto condition = translateExpression(firstOp(if_op.condition()), rewriter, symbolTable);
  bool else_branch = if_op->getNumRegions() == 3;
  auto new_if = rewriter.create<scf::IfOp>(if_op->getLoc(), condition, else_branch);

  std::unordered_map<std::string, Value> original_values;

  // THEN BRANCH (always exists)
  std::unordered_map<std::string, Value> then_values;
  std::function<void(const std::string &, Value, Value)>
      then_callback = [&](const std::string &name, Value oldValue, Value newValue)
  {
    then_values.insert({name, newValue});
    original_values.insert({name, oldValue});
  };
  rewriter.mergeBlocks(&getBlock(if_op.thenBranch()), new_if.thenBlock());
  for (auto &inner_op : llvm::make_early_inc_range(new_if.thenBlock()->getOperations()))
  {
    CustomSymbolTable::ScopeTy then_scope(symbolTable);
    translateStatement(inner_op, rewriter, symbolTable, nullptr, &then_callback);
  }

  // Reset values to original values
  for (const auto &p : then_values)
  {
    auto cur = symbolTable.lookup(p.first);
    symbolTable.insert(p.first, {cur.first, original_values.find(p.first)->second});
  }

  // ELSE BRANCH
  std::unordered_map<std::string, Value> else_values;
  std::function<void(const std::string &, Value, Value)>
      else_callback = [&](const std::string &name, Value oldValue, Value newValue)
  {
    else_values.insert({name, newValue});
    original_values.insert({name, oldValue});
  };
  if (else_branch)
  {
    rewriter.mergeBlocks(&getBlock(if_op.elseBranch().front()), new_if.elseBlock());
    for (auto &inner_op : llvm::make_early_inc_range(new_if.elseBlock()->getOperations()))
    {
      CustomSymbolTable::ScopeTy else_scope(symbolTable);
      translateStatement(inner_op, rewriter, symbolTable, nullptr, &else_callback);
    }
  }

  // Emit "MUX" statements & update symbolTable to refer to them
  // the values in here aren't useful since they're a mix, but it has the list of all update names
  auto updated_values = then_values;
  updated_values.insert(else_values.begin(), else_values.end());
  for (const auto &p : updated_values)
  {
    auto name = p.first;
    if (symbolTable.count(name) == 0)
    {
      // TODO: TranslateIfOp currently assumes no shadowing ever happens!
      continue; // this was a local variable inside the then/else branch and isn't currently in scope
    }
    auto type = symbolTable.lookup(name).first;
    auto updated_in_then = then_values.find(name) != then_values.end();
    auto condThenValue = Value();
    auto updated_in_else = else_values.find(name) != else_values.end();
    auto condElseValue = Value();
    if (updated_in_then)
    {
      condThenValue =
          rewriter.create<fhe::MultiplyOp>(if_op->getLoc(), ValueRange({condition, then_values.find(name)->second}));
    }
    if (updated_in_else)
    {
      condElseValue =
          rewriter.create<fhe::MultiplyOp>(if_op->getLoc(), ValueRange({condition, else_values.find(name)->second}));
    }

    if (updated_in_else && updated_in_then)
    {
      auto newValue = rewriter.create<fhe::AddOp>(if_op->getLoc(), ValueRange{condThenValue, condElseValue});
      symbolTable.insert(name, {type, newValue});
    }
    else if (updated_in_then)
    {
      symbolTable.insert(name, {type, condThenValue});
    }
    else if (updated_in_else)
    {
      symbolTable.insert(name, {type, condElseValue});
    }
  }
}

void translateVariableDeclarationOp(ast::VariableDeclarationOp vardecl_op,
                                    IRRewriter &rewriter,
                                    CustomSymbolTable &symbolTable)
{

  if (vardecl_op.value().empty())
  {
    emitError(vardecl_op.getLoc(), "Declarations that do not specify a value are currently not supported.");
    return;
  }
  // Get Name, Type and Value
  auto name = vardecl_op.name();
  auto type = vardecl_op.type();
  // TODO: Support decls without value by defining default values?
  auto value = translateExpression(firstOp(vardecl_op.value().front()), rewriter, symbolTable);
  if (value.getType() != type)
  {
    if (auto secret_type = type.cast<fhe::SecretType>())
    {
      if (auto const_op = value.getDefiningOp<arith::ConstantOp>())
      {
        // If this was initialized with a literal, we can coerce the type: //TODO: add checks before type coercion
        value = rewriter.create<fhe::ConstOp>(vardecl_op->getLoc(),
                                              type_to_attr(secret_type.getPlaintextType(), const_op.getValue()));
      }
    }
    else
    {
      emitError(vardecl_op->getLoc(), "Variable initialized with incompatible type.");
    }
  }
  value.setLoc(NameLoc::get(StringAttr::get(value.getContext(), name), value.getLoc()));
  // TODO: Somehow check that value and type are compatible
  (void)declare(name, type, value, symbolTable); // void cast to suppress "unused result" warning
}

void translateAssignmentOp(ast::AssignmentOp assignment_op,
                           IRRewriter &rewriter,
                           CustomSymbolTable &symbolTable,
                           AffineForOp *for_op,
                           std::function<void(const std::string &, Value, Value)> *if_op_callback)
{
  // Get Name, Type and Value
  auto value = translateExpression(firstOp(assignment_op.value()), rewriter, symbolTable);
  llvm::StringRef target_name = "INVALID_TARGET";

  auto &targetOp = firstOp(assignment_op.target());

  if (auto variable_op = llvm::dyn_cast<ast::VariableOp>(targetOp))
  {
    // If it's a variable,
    target_name = variable_op.name();
    Type target_type = symbolTable.lookup(target_name).first;
    if (if_op_callback)
    {
      if_op_callback->operator()(target_name.str(), symbolTable.lookup(target_name).second, value);
    }
    symbolTable.insert(target_name, {target_type, value});
  }
  else if (auto index_access = llvm::dyn_cast<ast::IndexAccessOp>(targetOp))
  {
    if (auto target_variable = llvm::dyn_cast<ast::VariableOp>(firstOp(index_access.target())))
    {
      // if this is an index access, we need to first insert an operation, then update table with that result value
      // instead, we need to insert an operation and then update the value
      target_name = target_variable.name();
      auto index = translateExpression(firstOp(index_access.index()), rewriter, symbolTable);
      value = rewriter
                  .create<tensor::InsertOp>(assignment_op->getLoc(), value, symbolTable.lookup(target_name).second, index);
      if (if_op_callback)
      {
        if_op_callback->operator()(target_name.str(), symbolTable.lookup(target_name).second, value);
      }
    }
    else
    {
      emitError(assignment_op.getLoc(),
                "Expected Index Access target to be a variable, got " + firstOp(index_access.target()).getName().getStringRef());
    }
  }
  else
  {
    emitError(assignment_op.target().getLoc(), "Got invalid assignment target!");
  }

  if (for_op)
  {
    // NOTE:
    //  THE BELOW DOESN'T WORK BECAUSE IT SEEMS LIKE WE CAN'T ADD ITER_ARGS TO AN EXISTING FOR_OP?
    //  SO FOR NOW WE JUST PUT EVERYTHING IN WHEN WE GENERATE A FOR OP AND LET THE CANONICALIZATION GET RID OF UNNEEDED ONES
    //  check if the symbol table still contains the symbol at the parent scope.
    //  If yes, then it's not loop local and we need to do some yield stuff!
    //  Next, we should check if it's already been added to the iter_args!
    //  by checking if one of the iter args is the same value as the one we get by looking up the old value
    //  Finally, if we ARE updating an existing iter arg, we need to find the existing yield stmt and change it
    //  otherwise, we can just emit a new yield at the end of the loop
    //  However, this might be BAD in terms of iterator stuff since we're currently in an llvm:: make early inc range thing
    //  iterating over all the ops nested in this for op!
    // emitError(assignment_op->getLoc(), "Currently, we do not handle writing to variables in for loops correctly");
    symbolTable.insert(target_name, {Type(), value});
  }
  else
  {
    symbolTable.insert(target_name, {Type(), value});
  }
}

void translateSimpleForOp(ast::SimpleForOp &simple_for_op,
                          IRRewriter &rewriter,
                          CustomSymbolTable &symbolTable)
{

  AffineForOp *new_for_ptr = nullptr;
  std::vector<std::pair<StringRef, std::pair<Type, Value>>> existing_vars;

  // Create a new scope
  {
    // This sets curScope in symbolTable to varScope
    CustomSymbolTable::ScopeTy for_scope(symbolTable);

    // Get every variable that exists and dump it as an iter args,
    // since we can't add them later, but ones that don't get used
    // are easily optimized away by --canonicalize
    std::vector<Value> iter_arg_values;
    // Note: this requires using the MarbleHE/llvm-project fork since upstream LLVM doesn't include this iterator
    for (auto it = symbolTable.mapBegin(); it != symbolTable.mapEnd(); ++it)
    {
      auto name = it->getFirst();
      auto type_and_value = symbolTable.lookup(name);
      existing_vars.emplace_back(name, type_and_value);
      iter_arg_values.push_back(type_and_value.second); // to have them conveniently for the rewriter.create call
    }

    // Create the affine for loop
    auto new_for = rewriter.create<AffineForOp>(simple_for_op->getLoc(),
                                                simple_for_op.start().getLimitedValue(),
                                                simple_for_op.end().getLimitedValue(),
                                                1, // step size
                                                iter_arg_values);
    new_for_ptr = &new_for;

    // Update the symboltable with the "local version" (iter arg) of all existing variables
    auto iter_args_it = new_for.getRegionIterArgs().begin();
    for (auto &var : existing_vars)
    {
      symbolTable.insert(var.first, {var.second.first, *iter_args_it++});
    }

    if (declare(simple_for_op.iv(), new_for.getInductionVar().getType(), new_for.getInductionVar(), symbolTable)
            .failed())
    {
      emitError(simple_for_op->getLoc(), "Declaration of for-loop IV failed!");
    }

    // Move ABC Operations over into the new for loop's entryBlock
    rewriter.setInsertionPointToStart(new_for.getBody());
    auto abc_block_it = simple_for_op.body().getOps<ast::BlockOp>();
    if (abc_block_it.begin() == abc_block_it.end() || ++abc_block_it.begin() != abc_block_it.end())
    {
      emitError(simple_for_op.getLoc(), "Expected exactly one Block inside function!");
    }
    else
    {
      auto abc_block = *abc_block_it.begin();
      if (abc_block->getNumRegions() != 1 || !abc_block.body().hasOneBlock())
      {
        emitError(abc_block.getLoc(), "ABC BlockOp must contain exactly one region and exactly one Block in that!");
      }
      else
      {
        llvm::iplist<Operation> oplist;
        auto &bb = *abc_block.body().getBlocks().begin();
        rewriter.mergeBlocks(&bb, new_for.getBody());
      }
    }

    // Finally, go through the block and translate each operation
    // It's the responsiblity of VariableAssignment to update the iterArgs, so we pass this operation along
    for (auto &op : llvm::make_early_inc_range(new_for.getBody()->getOperations()))
    {
      translateStatement(op, rewriter, symbolTable, &new_for);
    }

    // Yield all the iter args
    llvm::SmallVector<Value, 4> yield_values;
    for (auto &var : existing_vars)
    {
      yield_values.push_back(symbolTable.lookup(var.first).second);
    }
    rewriter.setInsertionPointToEnd(new_for.getBody());
    rewriter.create<AffineYieldOp>(simple_for_op->getLoc(), yield_values);

    // TODO: This happens to introduce a bunch of redundant yield/iter_args that canonicalize doesn't catch
    //   fix properly and/or add canonicalization where if yield is same as start value, then it's removed.
    //   This is a low priority issue, since after unrolling, these DO get canonicalized away.

  } // exit loop scope (by destroying the CustomSymbolTable::ScopeTy object)

  // Update the existing variables in the symboltable now that we've left the scope
  auto res_it = new_for_ptr->result_begin();
  auto var_it = existing_vars.begin();
  for (size_t i = 0; i < existing_vars.size(); i++)
  {
    symbolTable.insert(var_it->first, {var_it->second.first, *res_it});
    res_it++;
    var_it++;
  }
}

//}
// void translateForOp(ast::ForOp &for_op,
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
////  auto &test_op = *llvm::dyn_cast<ast::BlockOp>(firstOp(for_op.initializer())).getOps().begin();
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

/// Takes an abc AST statement operation and replaces it with appropriate SSA-style operations,
/// recursively dealing with nested operation.
/// \param op
/// \param rewriter
/// \param symbolTable
/// \param current_for_op if not null (default value), then this means we are currently creating this AffineForOp's body
/// \param if_op_callback if not null (default value), then this means we are currently a then/else branch
void translateStatement(Operation &op,
                        IRRewriter &rewriter,
                        CustomSymbolTable &symbolTable,
                        AffineForOp *current_for_op,
                        std::function<void(const std::string &, Value, Value)> *if_op_callback)
{
  rewriter.setInsertionPoint(&op);
  if (auto block_op = llvm::dyn_cast<ast::BlockOp>(op))
  {
    // TODO: Support nested ast::BlockOp in ast2ssa
    emitError(op.getLoc(), "Nested Blocks are not yet supported.");
  }
  else if (auto return_op = llvm::dyn_cast<ast::ReturnOp>(op))
  {
    if (return_op.getNumRegions() == 0)
    {
      rewriter.create<mlir::func::ReturnOp>(op.getLoc());
    }
    else if (return_op.getNumRegions() == 1)
    {
      auto &return_value_expr = firstOp(return_op.value().front());
      rewriter.create<mlir::func::ReturnOp>(op.getLoc(), translateExpression(return_value_expr, rewriter, symbolTable));
    }
    else
    {
      // TODO: Support multiple return values from ast::ReturnOp in ast2ssa
      emitError(op.getLoc(), "Returning multiple values in abc.return is not yet supported.");
    }
    rewriter.eraseOp(&op);
  }
  else if (auto assignment_op = llvm::dyn_cast<ast::AssignmentOp>(op))
  {
    translateAssignmentOp(assignment_op, rewriter, symbolTable, current_for_op, if_op_callback);
    rewriter.eraseOp(&op);
  }
  else if (auto variable_declaration_op = llvm::dyn_cast<ast::VariableDeclarationOp>(op))
  {
    translateVariableDeclarationOp(variable_declaration_op, rewriter, symbolTable);
    rewriter.eraseOp(&op);
  }
  else if (auto for_op = llvm::dyn_cast<ast::ForOp>(op))
  {
    // TODO: Support general ast::ForOp in ast2ssa
    emitError(op.getLoc(), "General For Statements are not yet supported.");
  }
  else if (auto if_op = llvm::dyn_cast<ast::IfOp>(op))
  {
    translateIfOp(if_op, rewriter, symbolTable);
    rewriter.eraseOp(&op);
  }
  else if (auto scf_yield_op = llvm::dyn_cast<scf::YieldOp>(op))
  {
    // Do nothing
    emitWarning(op.getLoc(), "Encountered scf.yield while translating ABC statements.");
  }
  else if (auto affine_yield_op = llvm::dyn_cast<AffineYieldOp>(op))
  {
    // do nothing
    emitWarning(op.getLoc(), "Encountered affine.yield while translating ABC statements.");
  }
  else if (auto simple_for_op = llvm::dyn_cast<ast::SimpleForOp>(op))
  {
    translateSimpleForOp(simple_for_op, rewriter, symbolTable);
    rewriter.eraseOp(&op);
  }
  else
  {
    emitError(op.getLoc(),
              "Unexpected Op encountered while translating ABC statements: " + op.getName().getStringRef());
  }
}

/// Takes an abc.function and replaces it with a builtin.func
/// This function deals with the function name, arguments, etc itself
/// but uses various helper functions to convert the statements in the body
/// \param f
/// \param rewriter
/// \param symbolTable
void convertFunctionOp2FuncOp(ast::FunctionOp &f,
                              IRRewriter &rewriter,
                              CustomSymbolTable &symbolTable)
{
  // Read the existing function arguments
  std::vector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::DictionaryAttr> namedArgs;
  std::vector<OpOperand> arguments;
  for (auto op : f.parameters().getOps<ast::FunctionParameterOp>())
  {
    auto param_attr = op.typeAttr();
    argTypes.push_back(param_attr.getValue()); // this is the stored type, getType() would be TypeAttr/StringAttr
    llvm::SmallVector<mlir::NamedAttribute> namedAttrList;
    auto dialect = param_attr.getValue().getDialect().getNamespace().str();
    namedAttrList.push_back(rewriter.getNamedAttr(dialect + "." + op.name().str(), op.typeAttr()));
    namedArgs.push_back(rewriter.getDictionaryAttr(namedAttrList));
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
  CustomSymbolTable::ScopeTy var_scope(symbolTable);
  for (auto pair : llvm::zip(f.getRegion(0).getOps<ast::FunctionParameterOp>(), entryBlock->getArguments()))
  {
    auto op = std::get<0>(pair);
    auto arg = std::get<1>(pair);
    auto param_name = op.nameAttr().getValue();
    if (failed(declare(param_name, Type(), arg, symbolTable)))
    {
      mlir::emitError(arg.getLoc(), "Cannot translate FunctionParameter " + param_name + ": name is already taken.");
    }
  }

  // Move ABC Operations over into the new function's entryBlock
  auto abc_block_it = f.body().getOps<ast::BlockOp>();
  if (abc_block_it.begin() == abc_block_it.end() || ++abc_block_it.begin() != abc_block_it.end())
  {
    emitError(f.getLoc(), "Expected exactly one abc.block inside abc.function!");
  }
  else
  {
    auto abc_block = *abc_block_it.begin();
    if (abc_block->getNumRegions() != 1 || !abc_block.body().hasOneBlock())
    {
      emitError(abc_block.getLoc(), "ABC BlockOp must contain exactly one region and exactly one block in that!");
    }
    else
    {
      llvm::iplist<Operation> oplist;
      auto &bb = *abc_block.body().getBlocks().begin();
      rewriter.mergeBlocks(&bb, entryBlock);
    }
  }

  // Now we can remove the original function
  rewriter.eraseOp(f);

  // Finally, go through the block and translate each operation
  for (auto &op : llvm::make_early_inc_range(entryBlock->getOperations()))
  {
    translateStatement(op, rewriter, symbolTable);
  }
}

/// Executed on each builtin.module (ModuleOp)
void heco::LowerASTtoSSAPass::runOnOperation()
{
  ConversionTarget target(getContext());
  target.addLegalDialect<fhe::FHEDialect, AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect>();
  target.addIllegalDialect<ast::ASTDialect>();

  auto module = getOperation();
  assert(module->getNumRegions() == 1 && "builtin.module should have exactly one region.");

  // Get the (default) block in the module's only region:
  auto &region = module->getRegion(0);
  if (!region.hasOneBlock())
  {
    emitError(region.getLoc(), "AST module must contain a region with exactly one block!");
  }
  auto &block = region.getBlocks().front();

  /// Rewriter used to perform all rewrites in this pass
  IRRewriter rewriter(&getContext());

  /// Symbol table used to translate variables into SSA
  CustomSymbolTable symbolTable;

  // using llvm::make_early_inc_range to avoid iterator invalidation since we delete the abc.op inside the call
  for (auto f : llvm::make_early_inc_range(block.getOps<ast::FunctionOp>()))
  {
    convertFunctionOp2FuncOp(f, rewriter, symbolTable);
  }
}

void heco::LowerASTtoSSAPass::getDependentDialects(DialectRegistry &registry) const
{

  registry.insert<fhe::FHEDialect,
                  mlir::AffineDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}