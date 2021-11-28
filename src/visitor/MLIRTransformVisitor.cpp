#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/MLIRTransformVisitor.h"

SpecialMLIRTransformVisitor::SpecialMLIRTransformVisitor(mlir::FloatAttr module,
                                                         mlir::MLIRContext &context) : builder(&context) {}

void SpecialMLIRTransformVisitor::visit(LiteralFloat &elem) {
  module = builder.getF32FloatAttr(elem.getValue());
}
