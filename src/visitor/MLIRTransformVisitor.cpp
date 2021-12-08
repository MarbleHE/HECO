#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/MLIRTransformVisitor.h"

#include "ABCDialect.h"

using namespace mlir;
using namespace abc;

SpecialMLIRTransformVisitor::SpecialMLIRTransformVisitor(mlir::FloatAttr module,
                                                         mlir::MLIRContext &context) : builder(&context) {}

void SpecialMLIRTransformVisitor::visit(LiteralDouble &elem) {
  attr = builder.getF32FloatAttr(elem.getValue());
}

void SpecialMLIRTransformVisitor::visit(VariableDeclaration &elem) {
  if (elem.hasTarget()) {
    auto id = elem.getTarget().getIdentifier();
    auto val = 5; //SpecialMLIRTransformVisitor::visit(elem.getValue());
    ABC_VariableDeclarationOp(id, val);
  }
}