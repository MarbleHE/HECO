
#include <ast_opt_mlir/visitor/abc_ast_to_mlir_visitor.h>
#include <Errors.h>

using namespace abc;

SpecialAbcAstToMlirVisitor::SpecialAbcAstToMlirVisitor(mlir::MLIRContext &ctx) : builder(&ctx) {
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
};

void SpecialAbcAstToMlirVisitor::visit(AbstractExpression &expr) {
  // TODO
}

void SpecialAbcAstToMlirVisitor::visit(Block &elem) {
  std::cout << "Visited block" << std::endl;
  // TODO: add block
  for (auto &s: elem.getStatements()) {
    std::cout << s.get().toJson() << std::endl;
    s.get().accept(*this);
  }
}

void SpecialAbcAstToMlirVisitor::visit(VariableDeclaration &elem) {
  std::cout << "Visited VariableDeclaration" << std::endl;

  if (!elem.hasTarget()) {
    throw stork::runtime_error("Variable declaration must have a target.");
  }
  std::string name = elem.getTarget().getIdentifier();

  mlir::Attribute attr;
  if (elem.getDatatype() == Datatype(Type::VOID)) {
    attr = builder.getF64FloatAttr(dynamic_cast<const LiteralDouble &>(elem.getValue()).getValue());
  }
  // TODO: add other data types
  // TODO: how do we evaluate expressions here?

  auto var_decl = builder.create<VariableDeclarationOp>(builder.getUnknownLoc(),  name, attr.getType(), 1);
//  var_decl->setOperand(0, builder.create<LiteralDoubleOp>(builder.getUnknownLoc(), attr));
  // TODO: find a way how to assign te value here
  module.push_back(var_decl);
  module.dump();
}

void SpecialAbcAstToMlirVisitor::visit(Return &elem) {
  // TODO
}
