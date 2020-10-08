#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/IndexAccess.h"

#include "ast_opt/ast/Literal.h"

// ================================================
// AST EXPRESSIONS
// ================================================

void SpecialTypeCheckingVisitor::visit(BinaryExpression &elem) {
  // check if both operands have the same data type
  elem.getLeft().accept(*this);
  auto lhsType = typesVisitedNodes.top();
  typesVisitedNodes.pop();

  elem.getRight().accept(*this);
  auto rhsType = typesVisitedNodes.top();
  typesVisitedNodes.pop();

  if (!areCompatibleDatatypes(lhsType, rhsType))
    throw std::runtime_error("Cannot apply operator (" + elem.getOperator().toString() + ") on operands of type "
                                 + enumToString(lhsType.getType()) + " and "
                                 + enumToString(rhsType.getType()) + ".");

  // store the datatype of the result: this is the result type of this expression
  expressionsDatatypeMap.insert_or_assign(elem.getUniqueNodeId(), lhsType);

  // check if any of the operands is secret -> result will be secret too
  auto resultIsSecret = (lhsType.getSecretFlag() || rhsType.getSecretFlag());
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), resultIsSecret);

  // save the type of this binary expression, required in case that this is nested
  typesVisitedNodes.push(Datatype(lhsType.getType(), resultIsSecret));
}

void SpecialTypeCheckingVisitor::visit(Call &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialTypeCheckingVisitor::visit(ExpressionList &elem) {
  ScopedVisitor::visit(elem);

  // check that the type of each expression in the expression list is the same
  auto firstExpression = typesVisitedNodes.top();
  typesVisitedNodes.pop();
  bool isSecret = firstExpression.getSecretFlag();
  Type expectedType = firstExpression.getType();
  while (!typesVisitedNodes.empty()) {
    auto curNode = typesVisitedNodes.top();
    Type curType = curNode.getType();
    isSecret |= curNode.getSecretFlag();
    typesVisitedNodes.pop();
    if (curType!=expectedType) {
      throw std::runtime_error("Values in ExpressionList must all be of the same type!");
    }
  }

  // push data type to parent as this can be a nested expression (e.g., {1, 2, 3} * var)
  typesVisitedNodes.push(Datatype(expectedType, isSecret));
}

void SpecialTypeCheckingVisitor::visit(FunctionParameter &elem) {
  ScopedVisitor::visit(elem);
  ScopedIdentifier scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getIdentifier());
  variablesDatatypeMap.insert_or_assign(scopedIdentifier, elem.getParameterType());
}

void SpecialTypeCheckingVisitor::visit(IndexAccess &elem) {
  size_t initial_size = typesVisitedNodes.size();

  ScopedVisitor::visit(elem);

  // an index access must always evaluate to an integer, i.e., all involved variables/literals must be integers too
  while (typesVisitedNodes.size() > initial_size) {
    auto currentType = typesVisitedNodes.top();
    typesVisitedNodes.pop();
    if (currentType.getType()!=Type::INT) {
      throw std::runtime_error("IndexAccess requires integers!");
    }
  }
}

void SpecialTypeCheckingVisitor::visit(LiteralBool &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::BOOL, false));
}

void SpecialTypeCheckingVisitor::visit(LiteralChar &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::CHAR, false));
}

void SpecialTypeCheckingVisitor::visit(LiteralInt &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::INT, false));
}

void SpecialTypeCheckingVisitor::visit(LiteralFloat &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::FLOAT, false));
}

void SpecialTypeCheckingVisitor::visit(LiteralDouble &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::DOUBLE, false));
}

void SpecialTypeCheckingVisitor::visit(LiteralString &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::STRING, false));
}

void SpecialTypeCheckingVisitor::visit(UnaryExpression &elem) {
  ScopedVisitor::visit(elem);

  auto operandType = typesVisitedNodes.top();
  typesVisitedNodes.pop();

  expressionsDatatypeMap.insert_or_assign(elem.getUniqueNodeId(), operandType);
  typesVisitedNodes.push(operandType);

  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), operandType.getSecretFlag());
}

void SpecialTypeCheckingVisitor::visit(Variable &elem) {
  ScopedVisitor::visit(elem);
  auto scopedIdentifier = variablesDatatypeMap.find(getCurrentScope().resolveIdentifier(elem.getIdentifier()));
  if (scopedIdentifier!=variablesDatatypeMap.end()) {
    typesVisitedNodes.push(scopedIdentifier->second);
  }
}


// ================================================
// AST STATEMENTS
// ================================================

void SpecialTypeCheckingVisitor::visit(Block &elem) {
  ScopedVisitor::visit(elem);
  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(For &elem) {
  ScopedVisitor::visit(elem);
  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(Function &elem) {
  ScopedVisitor::visit(elem);

  if (elem.getReturnType().getType()!=Type::VOID && returnExpressionTypes.empty()) {
    throw std::runtime_error("Return type specified (i.e., not void) but no return value found.");
  }

  // check if type and secretness of Return expression matches the one specified in the function's signature
  auto specifiedReturnDatatype = elem.getReturnType();
  for (auto &[t, literalValue] : returnExpressionTypes) {
    if (elem.getReturnType().getType()==Type::VOID) {
      throw std::runtime_error("Return value found in program although function's signature is declared as 'void'.");
    } else if (t.getType()!=specifiedReturnDatatype.getType()) {
      throw std::runtime_error("Type specified in function's signature does not match type of return statement.");
    } else if (!literalValue && t.getSecretFlag()!=specifiedReturnDatatype.getSecretFlag()) {
      throw std::runtime_error(
          "Secretness specified in function's signature does not match secretness of return statement. "
          "Note that if any of the involved operands of an expression are secret, the whole expression becomes secret.");
    }
  }

  returnExpressionTypes.clear();

  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(If &elem) {
  ScopedVisitor::enterScope(elem);

  elem.getCondition().accept(*this);
  typesVisitedNodes.pop();

  elem.getThenBranch().accept(*this);

  if (elem.hasElseBranch()) {
    elem.getElseBranch().accept(*this);
  }

  ScopedVisitor::exitScope();

  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);

  if (elem.hasValue()) {
    auto typeReturnExpr = typesVisitedNodes.top();
    typesVisitedNodes.pop();
    returnExpressionTypes.emplace_back(typeReturnExpr, isLiteral(elem.getValue()));
  }

  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(Assignment &elem) {
  ScopedVisitor::visit(elem);
  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(VariableDeclaration &elem) {
  ScopedVisitor::visit(elem);
  ScopedIdentifier scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getTarget().getIdentifier());
  variablesDatatypeMap.insert_or_assign(scopedIdentifier, elem.getDatatype());
  if (elem.hasValue()) {
    typesVisitedNodes.pop();
  }
  postStatementAction();
}


// ================================================
// OTHER METHODS
// ================================================

bool SpecialTypeCheckingVisitor::areCompatibleDatatypes(Datatype &first, Datatype &second) {
  return (first.getType()==second.getType());
}

Datatype SpecialTypeCheckingVisitor::getVariableDatatype(ScopedIdentifier &scopedIdentifier) {
  return variablesDatatypeMap.at(scopedIdentifier);
}

void SpecialTypeCheckingVisitor::postStatementAction() {
  if (!typesVisitedNodes.empty()) {
    throw std::runtime_error("Temporary stack was not cleaned up prior leaving statement! "
                             "Did you forget to pop() after retrieving element using top()?");
  }
}

Datatype SpecialTypeCheckingVisitor::getExpressionDatatype(AbstractExpression &expression) {
  if (expressionsDatatypeMap.count(expression.getUniqueNodeId()) > 0) {
    return expressionsDatatypeMap.at(expression.getUniqueNodeId());
  }
  throw std::runtime_error("Cannot get datatype of expression (" + expression.getUniqueNodeId() + ").");
}

bool SpecialTypeCheckingVisitor::isSecretTaintedNode(const std::string &uniqueNodeId) {
  if (secretTaintedNodes.count(uniqueNodeId)==0) {
    throw std::runtime_error("");
  }
  return secretTaintedNodes.at(uniqueNodeId);
}

const SecretTaintedNodesMap &SpecialTypeCheckingVisitor::getSecretTaintedNodes() const {
  return secretTaintedNodes;
}

SecretTaintedNodesMap &SpecialTypeCheckingVisitor::getSecretTaintedNodes() {
  return secretTaintedNodes;
}
