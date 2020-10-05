#include "ast_opt/visitor/Runtime/RuntimeVisitor.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/Variable.h"

#include "ast_opt/parser/Tokens.h"

SpecialRuntimeVisitor::SpecialRuntimeVisitor(AbstractNode &inputs, AbstractNode &outputs) {

}

AbstractExpression &SpecialRuntimeVisitor::getNextStackElement() {
  auto elem = intermedResult.top();
  intermedResult.pop();
  return elem.get();
}

void SpecialRuntimeVisitor::visit(BinaryExpression &elem) {
  ScopedVisitor::visit(elem);

  auto &rhsOperand = getNextStackElement();
  auto &lhsOperand = getNextStackElement();

  // TODO: Implement me!

}

void SpecialRuntimeVisitor::visit(UnaryExpression &elem) {
  ScopedVisitor::visit(elem);

  // TODO: Implement me!
}

void SpecialRuntimeVisitor::visit(Block &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Call &elem) {
  if (stork::getKeyword(elem.getIdentifier())==stork::reservedTokens::kw_rotate) {
    // arguments:
    //  (1) identifier
    //  (2) #steps to rotate ciphertext

    auto identifier = elem.getIdentifier();

    // TODO: Implement me!
  } else {
    throw std::runtime_error("Calls other than 'rotate(identifier, numSteps);' not supported yet!");
  }
}

void SpecialRuntimeVisitor::visit(ExpressionList &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(For &elem) {
  elem.getInitializer().accept(*this);

  // a helper method to check the value of the For loop's condition
  auto evaluateCondition = [&](AbstractExpression &expr) -> bool {
    expr.accept(*this);
    auto &result = getNextStackElement();
    if (auto conditionLiteralBool = dynamic_cast<LiteralBool *>(&result)) {
      return conditionLiteralBool->getValue();
    } else {
      throw std::runtime_error("For loop's condition must be ");
    }
  };

  // execute the For loop
  if (elem.hasCondition()) {
    while (evaluateCondition(elem.getCondition())) {
      if (elem.hasBody()) elem.getBody().accept(*this);
      if (elem.hasUpdate()) elem.getUpdate().accept(*this);
    }
  } else {
    throw std::runtime_error("For loops without a condition are not supported yet!");
  }
}

void SpecialRuntimeVisitor::visit(Function &elem) {
  throw std::runtime_error("Function statements not supported yet by RuntimeVisitor.");
//  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(If &elem) {
  ScopedVisitor::visit(elem);

  // get the If statement's condition
  elem.getCondition().accept(*this);
  auto &conditionResult = getNextStackElement();

  if (auto conditionLiteralBool = dynamic_cast<LiteralBool *>(&conditionResult)) {
    if (conditionLiteralBool->getValue()) {
      // visit "then" branch
      elem.getThenBranch().accept(*this);
    } else if (elem.hasElseBranch()) {
      // visit "else" branch if existent
      elem.getElseBranch().accept(*this);
    }
  } else {
    throw std::runtime_error("Condition of If statement must evaluate to a LiteralBool");
  }
}

void SpecialRuntimeVisitor::visit(IndexAccess &elem) {
  ScopedVisitor::visit(elem);
  // TODO: Implement me!
}

void SpecialRuntimeVisitor::visit(LiteralBool &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralChar &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralInt &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralFloat &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralDouble &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(LiteralString &elem) {
  ScopedVisitor::visit(elem);
  intermedResult.push(elem);
}

void SpecialRuntimeVisitor::visit(OperatorExpression &elem) {
  // TODO: OperatorExpressions still required? If yes, extend AbstractCiphertext + SealCiphertext with respective
  //  methods, test them, and make us of it here.
  throw std::runtime_error("UNIMPLEMENTED: RuntimeVisitor cannot executed OperatorExpressions yet.");
  //  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);
  // TODO: Implement me!
}

void SpecialRuntimeVisitor::visit(Assignment &elem) {
  ScopedVisitor::visit(elem);
  // TODO: Implement me!
}

void SpecialRuntimeVisitor::visit(VariableDeclaration &elem) {
  if (elem.hasValue()) elem.getValue().accept(*this);
  elem.accept(*this);
  auto &initializationValue = getNextStackElement();

  // TODO: Implement me!
  if (elem.getDatatype().getSecretFlag()) {
    // declaration of a secret variable, i.e., ciphertext

  } else {
    // declaration of a non-secret variable

  }
}

void SpecialRuntimeVisitor::visit(Variable &elem) {
  ScopedVisitor::visit(elem);
  // TODO: Implement me!
}
