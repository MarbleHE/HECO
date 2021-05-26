#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/CountOpsVisitor.h"
#include "ast_opt/ast/AbstractExpression.h"

SpecialCountOpsVisitor::SpecialCountOpsVisitor(AbstractNode &inputs) {
  _number_ops = 0;
}

void SpecialCountOpsVisitor::visit(AbstractNode &elem) {

}

void SpecialCountOpsVisitor::visit(BinaryExpression &elem) {
   _number_ops++;
}

int SpecialCountOpsVisitor::getNumberOps(){
  return _number_ops;
}

void SpecialCountOpsVisitor::executeAst(AbstractNode &rootNode) {
  try {
    rootNode.accept(*this);
  } catch (ReturnStatementReached &) {
    //TODO
  }
}
