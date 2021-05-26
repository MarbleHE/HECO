#include <ast_opt/utilities/Operator.h>
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/CountOpsVisitor.h"
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/BinaryExpression.h"

SpecialCountOpsVisitor::SpecialCountOpsVisitor(AbstractNode &inputs) {
  _number_ops = 0;
}

std::unique_ptr<AbstractValue> SpecialCountOpsVisitor::getNextStackElement() {
  auto elem = std::move(intermedResult.top());
  intermedResult.pop();
  return elem;
}

void SpecialCountOpsVisitor::visit(BinaryExpression &elem) {
  // ---- some helper methods -------------------------
  auto operatorEqualsAnyOf = [&elem](std::initializer_list<OperatorVariant> op) -> bool {
    return std::any_of(op.begin(), op.end(), [&elem](OperatorVariant op) { return elem.getOperator()==Operator(op); });
  };
  auto operatorEquals = [&elem](OperatorVariant op) -> bool { return elem.getOperator()==Operator(op); };
  // ---- end

  elem.getLeft().accept(*this);
  auto lhsOperand = getNextStackElement();

  elem.getRight().accept(*this);
  auto rhsOperand = getNextStackElement();

  // execute the binary operation
  if (operatorEqualsAnyOf({ADDITION, FHE_ADDITION})) {
    _number_ops++;
    _number_adds++;
  } else if (operatorEqualsAnyOf({SUBTRACTION, FHE_SUBTRACTION})) {
    _number_ops++;
  } else if (operatorEqualsAnyOf({MULTIPLICATION, FHE_MULTIPLICATION})) {
    _number_ops++;
  } else if (operatorEquals(DIVISION)) {
    _number_ops++;
  } else if (operatorEquals(MODULO)) {
    _number_ops++;
  } else if (operatorEquals(LOGICAL_AND)) {
    _number_ops++;
  } else if (operatorEquals(LOGICAL_OR)) {
    _number_ops++;
  } else if (operatorEquals(LESS)) {
    _number_ops++;
  } else if (operatorEquals(LESS_EQUAL)) {
    _number_ops++;
  } else if (operatorEquals(GREATER)) {
    _number_ops++;
  } else if (operatorEquals(GREATER_EQUAL)) {
    _number_ops++;
  } else if (operatorEquals(EQUAL)) {
    _number_ops++;
  } else if (operatorEquals(NOTEQUAL)) {
    _number_ops++;
  } else if (operatorEquals(BITWISE_AND)) {
    _number_ops++;
  } else if (operatorEquals(BITWISE_XOR)) {
    _number_ops++;
  } else if (operatorEquals(BITWISE_OR)) {
    _number_ops++;
  } else {
    throw std::runtime_error("Unknown binary operator encountered. Cannot continue!");
  }
}

void SpecialCountOpsVisitor::visit(AbstractNode &elem) {

}

int SpecialCountOpsVisitor::getNumberOps(){
  return _number_ops;
}

void SpecialCountOpsVisitor::executeAst(AbstractNode &rootNode) {
 // try {
    rootNode.accept(*this);
//  } catch (ReturnStatementReached &) {
    //TODO
 // }
}
