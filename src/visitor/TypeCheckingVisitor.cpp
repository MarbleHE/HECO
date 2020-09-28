#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast/BinaryExpression.h"
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

bool isCompatible(Datatype &first, Datatype &second) {
  return (first.getType()==second.getType());
}

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

  if (!isCompatible(lhsType, rhsType))
    throw std::runtime_error("Cannot apply operator (" + elem.getOperator().toString() + ") on operands of type "
                                 + enumToString(lhsType.getType()) + " and "
                                 + enumToString(rhsType.getType()) + ".");

  // store the datatype of the result: this is the type of this expression
  expressionsDatatypeMap.insert_or_assign(elem.getUniqueNodeId(), lhsType);

  // check if any of the operands is secret -> result will be secret too
  auto resultIsSecret = (lhsType.getSecretFlag() || rhsType.getSecretFlag());
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), resultIsSecret);
}

void SpecialTypeCheckingVisitor::visit(Call &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialTypeCheckingVisitor::visit(ExpressionList &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialTypeCheckingVisitor::visit(FunctionParameter &elem) {
  ScopedVisitor::visit(elem);
  ScopedIdentifier scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getIdentifier());
  variablesDatatypeMap.insert_or_assign(scopedIdentifier, elem.getParameterType());
}

void SpecialTypeCheckingVisitor::visit(IndexAccess &elem) {
  int initial_size = typesVisitedNodes.size();
  // an index access must always evaluate to an integer, i.e., all involved variables/literals must be integers too
  ScopedVisitor::visit(elem);

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
  checkStatementFinished();
}

void SpecialTypeCheckingVisitor::visit(For &elem) {
  ScopedVisitor::visit(elem);
  checkStatementFinished();
}

void SpecialTypeCheckingVisitor::visit(Function &elem) {
  ScopedVisitor::visit(elem);
  checkStatementFinished();
}

void SpecialTypeCheckingVisitor::visit(If &elem) {
  ScopedVisitor::visit(elem);
  checkStatementFinished();
}

void SpecialTypeCheckingVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);
  checkStatementFinished();
}

void SpecialTypeCheckingVisitor::visit(Assignment &elem) {
  ScopedVisitor::visit(elem);
  checkStatementFinished();
}

void SpecialTypeCheckingVisitor::visit(VariableDeclaration &elem) {
  ScopedVisitor::visit(elem);
  ScopedIdentifier scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getTarget().getIdentifier());
  variablesDatatypeMap.insert_or_assign(scopedIdentifier, elem.getDatatype());
  if (elem.hasValue()) typesVisitedNodes.pop();
  checkStatementFinished();
}


// ================================================
// OTHER METHODS
// ================================================


Datatype &SpecialTypeCheckingVisitor::getVariableDatatype(ScopedIdentifier &scopedIdentifier) {
  return variablesDatatypeMap.at(scopedIdentifier);
}

void SpecialTypeCheckingVisitor::checkStatementFinished() {
  if (!typesVisitedNodes.empty()) {
    throw std::runtime_error("Temporary stack was not cleaned up prior leaving statement! "
                             "Did you forget to pop() after retrieving element using top()?");
  }
}

