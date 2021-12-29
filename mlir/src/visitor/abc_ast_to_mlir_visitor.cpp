
#include <ast_opt_mlir/visitor/abc_ast_to_mlir_visitor.h>
#include <Errors.h>

using namespace abc;

SpecialAbcAstToMlirVisitor::SpecialAbcAstToMlirVisitor(mlir::MLIRContext &ctx) : builder(&ctx) {
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
}

void SpecialAbcAstToMlirVisitor::add_op(mlir::Operation *op) {
  if (block->empty()) {
    block->push_back(op);
  } else {
    module.push_back(op);
  }
}

void SpecialAbcAstToMlirVisitor::add_recursive_result_to_region(AbstractNode &node, mlir::Region &region) {
  mlir::Block *block = new mlir::Block();
  region.push_back(block);
  recursive_visit(node, block);
}

mlir::Type SpecialAbcAstToMlirVisitor::translate_type(Datatype abc_type) {
  // TODO (Miro): For some reason, there are no get*Type functions for Bool, Char, String
  // TODO (Miro): Is the none type the one corresponding to void?
  if (abc_type == Datatype(Type::BOOL))
    return builder.getBoolAttr(false).getType();
  else if (abc_type == Datatype(Type::CHAR))
    return builder.getStringAttr(mlir::Twine('.')).getType();
  else if (abc_type == Datatype(Type::DOUBLE))
    return builder.getF64Type();
  else if (abc_type == Datatype(Type::FLOAT))
    return builder.getF32Type();
  else if (abc_type == Datatype(Type::INT))
    return builder.getI64Type();
  else if (abc_type == Datatype(Type::STRING))
    return builder.getStringAttr(mlir::Twine("..")).getType();
  else if (abc_type == Datatype(Type::VOID))
    return builder.getNoneType();
  else
    throw stork::runtime_error("Unknown ABC type");
}

void SpecialAbcAstToMlirVisitor::recursive_visit(AbstractNode &node, mlir::Block *childBlock) {
  // Store current block and use a fresh one for the recursive child visit
  mlir::Block *parentBlock = block;
  block = childBlock;
  if (auto expr = dynamic_cast<AbstractExpression *>(&node)) {
    visit(*expr);
  } else if (auto stmt = dynamic_cast<AbstractStatement *>(&node)) {
    visit(*stmt);
  } else {
    throw stork::runtime_error("Unknown subclass of AbstractNode.");
  }
  block = parentBlock;
}

void SpecialAbcAstToMlirVisitor::visit(AbstractExpression &expr) {
  if (auto variable = dynamic_cast<Variable *>(&expr)) {
    visit(*variable);
  } else if (auto idxAccess = dynamic_cast<IndexAccess *>(&expr)) {
    visit(*idxAccess);
  } else if (auto fnParam = dynamic_cast<FunctionParameter *>(&expr)) {
    visit(*fnParam);
  } else if (auto binExpr = dynamic_cast<BinaryExpression *>(&expr)) {
    visit(*binExpr);
  } else if (auto opExpr = dynamic_cast<OperatorExpression *>(&expr)) {
    visit(*opExpr);
  } else if (auto unaryExpr = dynamic_cast<UnaryExpression *>(&expr)) {
    visit(*unaryExpr);
  } else if (auto call = dynamic_cast<Call *>(&expr)) {
    visit(*call);
  } else if (auto exprList = dynamic_cast<ExpressionList *>(&expr)) {
    visit(*exprList);
  } else if (auto litBool = dynamic_cast<LiteralBool *>(&expr)) {
    visit(*litBool);
  } else if (auto litChr = dynamic_cast<LiteralChar *>(&expr)) {
    visit(*litChr);
  } else if (auto litDbl = dynamic_cast<LiteralDouble *>(&expr)) {
    visit(*litDbl);
  } else if (auto litFlt = dynamic_cast<LiteralFloat *>(&expr)) {
    visit(*litFlt);
  } else if (auto litInt = dynamic_cast<LiteralInt *>(&expr)) {
    visit(*litInt);
  } else if (auto litStr = dynamic_cast<LiteralString *>(&expr)) {
    visit(*litStr);
  } else if (auto ternOp = dynamic_cast<TernaryOperator *>(&expr)) {
    visit(*ternOp);
  } else {
    throw stork::runtime_error("Unknown AbstractExpression type occured in ABC to MLIR translation.");
  }
}

void SpecialAbcAstToMlirVisitor::visit(Assignment &elem) {
  auto assignOp = builder.create<AssignmentOp>(builder.getUnknownLoc());

  // Target
  mlir::Block *tarBlock = new mlir::Block();
  assignOp.target().push_back(tarBlock);
  recursive_visit(elem.getTarget(), tarBlock);

  // Value
  mlir::Block *valBlock = new mlir::Block();
  assignOp.value().push_back(valBlock);
  recursive_visit(elem.getValue(), valBlock);

  // Add new assignment operation
  add_op(assignOp);
}

void SpecialAbcAstToMlirVisitor::visit(BinaryExpression &elem) {
  auto opAttr = builder.getStringAttr(llvm::Twine(elem.getOperator().toString()));
  auto binExpr = builder.create<BinaryExpressionOp>(builder.getUnknownLoc(), opAttr);

  // Add LHS
  mlir::Block *lhsBlock = new mlir::Block();
  binExpr.left().push_back(lhsBlock);
  recursive_visit(elem.getLeft(), lhsBlock);

  // Add RHS
  mlir::Block *rhsBlock = new mlir::Block();
  binExpr.right().push_back(rhsBlock);
  recursive_visit(elem.getRight(), rhsBlock);

  // Add binary operation
  add_op(binExpr);
}

void SpecialAbcAstToMlirVisitor::visit(Block &elem) {
  // Temporarily collect statements in a different block
  mlir::Block *prevBlock = block;
  block = new mlir::Block();

  // Iterate over all statements
  for (auto &s : elem.getStatements()) {
    s.get().accept(*this);
  }

  // Add abc.block op
  auto blockOp = builder.create<BlockOp>(builder.getUnknownLoc());
  blockOp.getRegion().push_back(block);
  prevBlock->push_back(blockOp);
  block = prevBlock;
}

// TODO: Functions are not yet supported by the frontend
void SpecialAbcAstToMlirVisitor::visit(Call &elem) {
  auto fnName = builder.getStringAttr(llvm::Twine(elem.getIdentifier()));
  auto callOp = builder.create<CallOp>(builder.getUnknownLoc(), fnName);

  mlir::Block *argBlock;
  for (auto argExpr : elem.getArguments()) {
    argBlock = new mlir::Block();
    callOp.arguments().push_back(argBlock);
    recursive_visit(argExpr, argBlock);
  }
}

void SpecialAbcAstToMlirVisitor::visit(For &elem) {
  auto forOp = builder.create<ForOp>(builder.getUnknownLoc());

  // Convert initializer
  add_recursive_result_to_region(elem.getInitializer(), forOp.initializer());

  // Convert condition
  add_recursive_result_to_region(elem.getCondition(), forOp.condition());

  // Convert update
  add_recursive_result_to_region(elem.getUpdate(), forOp.update());

  // Convert body
  add_recursive_result_to_region(elem.getBody(), forOp.body());

  // Add for operation
  add_op(forOp);
}

void SpecialAbcAstToMlirVisitor::visit(Function &elem) {
  auto fnName = builder.getStringAttr(llvm::Twine(elem.getIdentifier()));
  auto type = translate_type(elem.getReturnType());
  auto fnOp = builder.create<FunctionOp>(builder.getUnknownLoc(), fnName, type);

  // Add parameters
  for (auto param : elem.getParameters()) {
    add_recursive_result_to_region(param, fnOp.parameters());
  }

  // Add body
  add_recursive_result_to_region(elem.getBody(), fnOp.body());

  // Add function to module (XXX: this makes the assumption that there are no nested functions...)
  module.push_back(fnOp);
}

void SpecialAbcAstToMlirVisitor::visit(FunctionParameter &elem) {
  auto fnParamName = builder.getStringAttr(llvm::Twine(elem.getIdentifier()));
  auto fnParamType = translate_type(elem.getParameterType());
  auto fnParamOp = builder.create<FunctionParameterOp>(builder.getUnknownLoc(), fnParamName, fnParamType);

  // add function parameter to current block
  add_op(fnParamOp);
}

void SpecialAbcAstToMlirVisitor::visit(If &elem) {
  auto ifOp = builder.create<IfOp>(builder.getUnknownLoc(), elem.hasElseBranch() ? 1 : 0);

  // Add condition
  add_recursive_result_to_region(elem.getCondition(), ifOp.condition());

  // Add then branch
  add_recursive_result_to_region(elem.getThenBranch(), ifOp.thenBranch());

  // Add else branch if present.
  if (elem.hasElseBranch()) {
    // Note that MLIR would support multiple else (if) branches, but the ABC AST only supports one.
    add_recursive_result_to_region(elem.getElseBranch(), ifOp.elseBranch().front());
  }

  // Add if condition
  add_op(ifOp);
}

// TODO (Miro): untested, first need to fix vectors in the Python Frontend
void SpecialAbcAstToMlirVisitor::visit(IndexAccess &elem) {
  auto idxAccessOp = builder.create<IndexAccessOp>(builder.getUnknownLoc());

  // Add target
  add_recursive_result_to_region(elem.getTarget(), idxAccessOp.target());

  // Add index expression
  add_recursive_result_to_region(elem.getIndex(), idxAccessOp.index());

  // Add index access operation
  add_op(idxAccessOp);

  block->dump();
}

void SpecialAbcAstToMlirVisitor::visit(LiteralBool &elem) {
  auto bval = builder.getBoolAttr(elem.getValue());
  block->push_back(builder.create<LiteralBoolOp>(builder.getUnknownLoc(), bval));
}

void SpecialAbcAstToMlirVisitor::visit(LiteralChar &elem) {
  auto cval = builder.getStringAttr(llvm::Twine(elem.getValue()));
  block->push_back(builder.create<LiteralCharOp>(builder.getUnknownLoc(), cval));
}

void SpecialAbcAstToMlirVisitor::visit(LiteralInt &elem) {
  auto i64val = builder.getI64IntegerAttr(elem.getValue());
  block->push_back(builder.create<LiteralIntOp>(builder.getUnknownLoc(), i64val));
}

void SpecialAbcAstToMlirVisitor::visit(LiteralFloat &elem) {
  auto f32val = builder.getF32FloatAttr(elem.getValue());
  block->push_back(builder.create<LiteralFloatOp>(builder.getUnknownLoc(), f32val));
}

void SpecialAbcAstToMlirVisitor::visit(LiteralDouble &elem) {
  auto f64val = builder.getF64FloatAttr(elem.getValue());
  block->push_back(builder.create<LiteralDoubleOp>(builder.getUnknownLoc(), f64val));
}

void SpecialAbcAstToMlirVisitor::visit(LiteralString &elem) {
  auto sval = builder.getStringAttr(llvm::Twine(elem.getValue()));
  block->push_back(builder.create<LiteralStringOp>(builder.getUnknownLoc(), sval));
}

// TODO (Miro): Untested
void SpecialAbcAstToMlirVisitor::visit(OperatorExpression &elem) {
  auto opAttr = builder.getStringAttr(llvm::Twine(elem.getOperator().toString()));
  auto opExpr = builder.create<OperatorExpressionOp>(builder.getUnknownLoc(), opAttr, elem.getOperands().size());

  // Add all operands as regions
  int i = 1; // the i = 0 region is not used for operands
  mlir::Block *operandBlock;
  for (auto operand : elem.getOperands()) {
    add_recursive_result_to_region(operand, opExpr.getRegion(i));
    ++i;
  }

  // Add new operator expression
  add_op(opExpr);
}

void SpecialAbcAstToMlirVisitor::visit(Return &elem) {
  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc(), elem.hasValue() ? 1 : 0);

  // Add returned expression
  if (elem.hasValue()) {
    // Note that the frontend currently only supports returning a single expression
    add_recursive_result_to_region(elem.getValue(), returnOp.value().front());
  }

  // Add return op
  add_op(returnOp);
}

void SpecialAbcAstToMlirVisitor::visit(AbstractStatement &stmt) {
  if (auto assign = dynamic_cast<Assignment *>(&stmt)) {
    visit(*assign);
  } else if (auto block = dynamic_cast<Block *>(&stmt)) {
    visit(*block);
  } else if (auto forStmt = dynamic_cast<For *>(&stmt)) {
    visit(*forStmt);
  } else if (auto fnStmt = dynamic_cast<Function *>(&stmt)) {
    visit(*fnStmt);
  } else if (auto ifStmt = dynamic_cast<If *>(&stmt)) {
    visit(*ifStmt);
  } else if (auto returnStmt = dynamic_cast<Return *>(&stmt)) {
    visit(*returnStmt);
  } else if (auto varDecl = dynamic_cast<VariableDeclaration *>(&stmt)) {
    visit(*varDecl);
  } else {
    throw stork::runtime_error("Unknown AbstractExpression type occured in ABC to MLIR translation.");
  }
}

void SpecialAbcAstToMlirVisitor::visit(UnaryExpression &elem) {
  auto opAttr = builder.getStringAttr(llvm::Twine(elem.getOperator().toString()));
  auto unExpr = builder.create<UnaryExpressionOp>(builder.getUnknownLoc(), opAttr);

  // Add operand
  add_recursive_result_to_region(elem.getOperand(), unExpr.operand());

  // Add unary expression operation
  add_op(unExpr);
}

void SpecialAbcAstToMlirVisitor::visit(VariableDeclaration &elem) {
  if (!elem.hasTarget()) {
    throw stork::runtime_error("Variable declaration must have a target.");
  }
  std::string name = elem.getTarget().getIdentifier();

  mlir::Block *targetBlock = new mlir::Block();
  recursive_visit(elem.getValue(), targetBlock);

  auto targetAttr = targetBlock->front().getAttr("value");
  auto varDeclOp = builder.create<VariableDeclarationOp>(builder.getUnknownLoc(),  name, targetAttr.getType(), 1);
  varDeclOp.getRegion(0).push_back(targetBlock);
  add_op(varDeclOp);
}

void SpecialAbcAstToMlirVisitor::visit(Variable &elem) {
  auto varOp = builder.create<VariableOp>(builder.getUnknownLoc(), elem.getIdentifier());
  block->push_back(varOp);
}
