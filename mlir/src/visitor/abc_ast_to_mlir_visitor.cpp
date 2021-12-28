
#include <ast_opt_mlir/visitor/abc_ast_to_mlir_visitor.h>
#include <Errors.h>

using namespace abc;

SpecialAbcAstToMlirVisitor::SpecialAbcAstToMlirVisitor(mlir::MLIRContext &ctx) : builder(&ctx) {
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
}

void SpecialAbcAstToMlirVisitor::add_op(mlir::Operation *op) {
  if (block->empty()) {
    block->push_back(op);
    block->dump();
  } else {
    module.push_back(op);
    module.dump();
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
  } else if (auto litBool = dynamic_cast<LiteralBool *>(&expr)) {
    visit(*litBool);
  } else if (auto ternOp = dynamic_cast<TernaryOperator *>(&expr)) {
    visit(*ternOp);
  } else {
    throw stork::runtime_error("Unknown AbstractExpression type occured in ABC to MLIR translation.");
  }
}

void SpecialAbcAstToMlirVisitor::visit(Block &elem) {
  std::cout << "Visited block" << std::endl;
  auto newBlockPtr = std::make_unique<mlir::Block>();
  // TODO: add block
//  block = newBlockPtr.get();
  for (auto &s: elem.getStatements()) {
    std::cout << s.get().toJson() << std::endl;
    s.get().accept(*this);
  }
//  module.getRegion().push_back(newBlockPtr.get());
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
  std::cout << "MLIR Translation Visitor: visited LiteralDouble" << std::endl;
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

//  mlir::Block *targetBlock = new mlir::Block();
//  recursive_visit(elem.getValue(), targetBlock);
//
//  auto targetExpr = targetBlock->;
//  auto varDeclOp = builder.create<VariableDeclarationOp>(builder.getUnknownLoc(),  name, attr.getType(), 1);
//  add_op(varDeclOp);
//
//  mlir::Attribute attr;
//  if (elem.getDatatype() == Datatype(Type::VOID)) {
//    attr = builder.getF64FloatAttr(dynamic_cast<const LiteralDouble &>(elem.getValue()).getValue());
//  }
//  // TODO: replace with recursive call to visit
//  // TODO: how do we evaluate expressions here?
//
//  auto var_decl = builder.create<VariableDeclarationOp>(builder.getUnknownLoc(),  name, attr.getType(), 1);
//  mlir::OperationState result = mlir::OperationState(builder.getUnknownLoc(), name);
//  result.addRegion();
//  auto assign = builder.create<LiteralDoubleOp>(builder.getUnknownLoc(), builder.getF64FloatAttr(dynamic_cast<const LiteralDouble &>(elem.getValue()).getValue()));
////  var_decl->getBlock()->push_back(assign);
////  result.regions.front()->push_back(assign->getBlock());
////  auto op = builder.createOperation(result);
//  // TODO: find a way how to assign te value here
//  block->push_back(assign);
//  block->push_back(var_decl);
//  block->dump();
}

void SpecialAbcAstToMlirVisitor::visit(Variable &elem) {
  std::cout << "MLIR Translation Visitor: visited Variable" << std::endl;
  auto varOp = builder.create<VariableOp>(builder.getUnknownLoc(), elem.getIdentifier());
  block->push_back(varOp);
}
