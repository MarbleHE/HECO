
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
  // TODO
}

void SpecialAbcAstToMlirVisitor::visit(BinaryExpression &elem) {
  auto opAttr = builder.getStringAttr(llvm::Twine(elem.getOperator().toString()));
  auto binExpr = builder.create<BinaryExpressionOp>(builder.getUnknownLoc(), opAttr);

  // Add LHS
  mlir::Block *lhsBlock = new mlir::Block();
  binExpr.getRegion(0).push_back(lhsBlock);
  recursive_visit(elem.getLeft(), lhsBlock);

  // Add RHS
  mlir::Block *rhsBlock = new mlir::Block();
  binExpr.getRegion(1).push_back(rhsBlock);
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
    std::cout << s.get().toJson() << std::endl;
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
  mlir::Block *initBlock = new mlir::Block();
  forOp.initializer().push_back(initBlock);
  recursive_visit(elem.getInitializer(), initBlock);

  // Convert condition
  mlir::Block *condBlock = new mlir::Block();
  forOp.condition().push_back(condBlock);
  recursive_visit(elem.getCondition(), condBlock);

  // Convert update
  mlir::Block *updateBlock = new mlir::Block();
  forOp.update().push_back(updateBlock);
  recursive_visit(elem.getUpdate(), updateBlock);

  // Convert body
  mlir::Block *bodyBlock = new mlir::Block();
  forOp.body().push_back(bodyBlock);
  recursive_visit(elem.getBody(), bodyBlock);

  // Add for operation
  add_op(forOp);

  module->dump();
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

void SpecialAbcAstToMlirVisitor::visit(Return &elem) {
  auto returnOp = builder.create<ReturnOp>(builder.getUnknownLoc(), 1);
  mlir::Block *retBlock = new mlir::Block();
  returnOp.getRegion(0).push_back(retBlock);
  recursive_visit(elem.getValue(), retBlock);
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
