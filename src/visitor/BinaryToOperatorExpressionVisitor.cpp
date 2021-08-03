#include "ast_opt/visitor/BinaryToOperatorExpressionVisitor.h"

//TODO: WRITE TESTS FOR THIS VISITOR

void SpecialBinaryToOperatorExpressionVisitor::visit(BinaryExpression &elem) {

  std::vector<std::unique_ptr<AbstractExpression>> operands;
  operands.emplace_back(elem.takeLeft());
  operands.emplace_back(elem.takeRight());
  replacement_expr = std::make_unique<OperatorExpression>(elem.getOperator(), std::move(operands));

  // TODO: Now, the visit call from which this was called (i.e. visit parent of elem) needs to replace elem with this new expression
  // TODO: Even better: This function should be able to replace it automagically!
  if(elem.hasParent()) {
    //elem.getParent().replaceChild(elem, std::move(replacement_expr));
  } else {
    // elem is the root => wrapper needs to handle this.
    // TODO: encapsulate this visitor inside a wrapper class that replaces the root!
  }
}

//// int x = 1 + 2; => int x = null + null; ....  replacement_expr +(1,2) => int x = +(1,2)
//void SpecialBinaryToOperatorExpressionVisitor::visit(AbstractNode &elem) {
//
//  for (auto &c: elem) {
//    c.accept(*this);
//    if (replacement_expr) {
//      // Replace c with the replacement expression!
//      //TODO: Can't replace a random child with another since we don't have a neat way to do this!
//      //      Try implementing some kind of interface (with type checking at runtime) for this in AbstractNode
//      //      This probably requires adding a second set of iterators or something similar :(
//      //c = std::move(replacement_expr);
//      //c.getOwners
//
//      // N.B. do not need to zero out replacement_expr since std::move already does that!
//    }
//  }
//}

//// Example to show what it would look like if we knew types
//void SpecialBinaryToOperatorExpressionVisitor::visit(Assignment &elem) {
//  elem.getTarget().accept(*this);
//  elem.getValue().accept(*this);
//  if (replacement_expr) {
//    elem.setValue(std::move(replacement_expr));
//  }
//}