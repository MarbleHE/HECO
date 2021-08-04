#include "ast_opt/visitor/BinaryToOperatorExpressionVisitor.h"

//TODO: WRITE TESTS FOR THIS VISITOR

void SpecialBinaryToOperatorExpressionVisitor::visit(BinaryExpression &elem) {

  elem.getLeft().accept(*this);
  elem.getRight().accept(*this);
  std::vector<std::unique_ptr<AbstractExpression>> operands;
  operands.emplace_back(elem.takeLeft());
  operands.emplace_back(elem.takeRight());
  replacement_expr = std::make_unique<OperatorExpression>(elem.getOperator(), std::move(operands));

  if (elem.hasParent()) {
    elem.getParent().replaceChild(elem, std::move(replacement_expr));
  } else {
    // elem is the root => wrapper needs to handle this.
    // TODO: encapsulate this visitor inside a wrapper class that replaces the root!
    //  This is a low priority task, since the parser never creates free standing expressions!
  }
}
