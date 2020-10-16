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
  typesVisitedNodes.emplace(lhsType.getType(), resultIsSecret);
}

void SpecialTypeCheckingVisitor::visit(Call &elem) {
  ScopedVisitor::visit(elem);
  // TODO: If any of the parameters passed to the called function are secret, then this Call would become secret too
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
  // save initial size to avoid making strong assumption that we only have a target and an index
  size_t initial_size = typesVisitedNodes.size();

  ScopedVisitor::visit(elem);

  // first, check the index: an index access must always evaluate to an integer
  while (typesVisitedNodes.size() - 1 > initial_size) {
    auto currentType = typesVisitedNodes.top();
    typesVisitedNodes.pop();
    if (currentType.getType()!=Type::INT) {
      throw std::runtime_error("IndexAccess requires integers!");
    }
  }

  // now retrieve the identifier
  auto identifierVariableType = typesVisitedNodes.top();
  typesVisitedNodes.pop();

  // update secret tainting of this node
  auto targetIsSecretTainted = secretTaintedNodes.at(elem.getTarget().getUniqueNodeId());
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), targetIsSecretTainted);

  // this is pushed to the stack for consistency as the caller expects the return type on the stack, however, we as the
  // variable this IndexAccess refers to has to be declared before we already know its datatype
  typesVisitedNodes.push(identifierVariableType);
}

void SpecialTypeCheckingVisitor::visit(LiteralBool &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::BOOL, false));
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), false);
}

void SpecialTypeCheckingVisitor::visit(LiteralChar &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::CHAR, false));
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), false);
}

void SpecialTypeCheckingVisitor::visit(LiteralInt &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::INT, false));
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), false);
}

void SpecialTypeCheckingVisitor::visit(LiteralFloat &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::FLOAT, false));
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), false);
}

void SpecialTypeCheckingVisitor::visit(LiteralDouble &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::DOUBLE, false));
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), false);
}

void SpecialTypeCheckingVisitor::visit(LiteralString &elem) {
  ScopedVisitor::visit(elem);
  typesVisitedNodes.push(Datatype(Type::STRING, false));
  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), false);
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
  auto scopedIdentifier = variablesDatatypeMap.find(getCurrentScope().resolveIdentifier(elem.getIdentifier()));
  if (scopedIdentifier!=variablesDatatypeMap.end()) {
    typesVisitedNodes.push(scopedIdentifier->second);
    secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), scopedIdentifier->second.getSecretFlag());
  } else {
    throw std::runtime_error(
        "Not datatype information found for variable (" + elem.getIdentifier() + "). "
            + "Did you forget to initialize it?");
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
  // special treatment for For loops: we need to visit the children of the initializer/update blocks separately as we
  // do not want to create a new scope when visiting them (otherwise variables declared in initializer will not be
  // accessible in condition and update)
  if (auto forStatement = dynamic_cast<For *>(&elem)) {
    // call visitChildren directly on the initializer block, otherwise this would create a new scope but that's wrong!
    visitChildren(forStatement->getInitializer());

    forStatement->getCondition().accept(*this);
    // we need to pop the stack as the condition is an expression unlike the initializer and update statement(s)
    typesVisitedNodes.pop();

    // call visitChildren directly on the update block, otherwise this would create a new scope but that's wrong!
    visitChildren(forStatement->getUpdate());

    forStatement->getBody().accept(*this);
  }
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
  bool anythingIsTainted = isSecretTaintedNode(elem.getCondition().getUniqueNodeId());

  elem.getThenBranch().accept(*this);

  anythingIsTainted = anythingIsTainted || isSecretTaintedNode(elem.getThenBranch().getUniqueNodeId());

  if (elem.hasElseBranch()) {
    elem.getElseBranch().accept(*this);
    anythingIsTainted = anythingIsTainted || isSecretTaintedNode(elem.getElseBranch().getUniqueNodeId());
  }

  ScopedVisitor::exitScope();

  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), anythingIsTainted);

  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);

  bool isTainted = false;
  if (elem.hasValue()) {
    auto typeReturnExpr = typesVisitedNodes.top();
    typesVisitedNodes.pop();
    returnExpressionTypes.emplace_back(typeReturnExpr, isLiteral(elem.getValue()));
    isTainted = isSecretTaintedNode(elem.getValue().getUniqueNodeId());
  }

  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), isTainted);

  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(Assignment &elem) {
  // No need to visit the left-hand side of an assignment as it does not tell anything about its datatype
  registerBeforeVisitChildren();
  elem.getValue().accept(*this);
  // The declaration of this identifier already provided us the information about its type, hence we can just discard
  // the datatype in typesVisitedNodes.
  discardChildrenDatatypes();

  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), isSecretTaintedNode(elem.getValue().getUniqueNodeId()));

  postStatementAction();
}

void SpecialTypeCheckingVisitor::visit(VariableDeclaration &elem) {
  // visiting the left-hand side of the declaration is not necessary and causes issues as the datatype for this variable
  // is not known yet (see few lines below) but the identifier must be registered before we can store the datatype
  getCurrentScope().addIdentifier(elem.getTarget().getIdentifier());
  ScopedIdentifier scopedIdentifier = getCurrentScope().resolveIdentifier(elem.getTarget().getIdentifier());
  variablesDatatypeMap.insert_or_assign(scopedIdentifier, elem.getDatatype());

  bool isTainted = false;
  if (elem.hasValue()) {
    elem.getValue().accept(*this);
    typesVisitedNodes.pop();
    isTainted = isSecretTaintedNode(elem.getValue().getUniqueNodeId());
  }

  secretTaintedNodes.insert_or_assign(elem.getUniqueNodeId(), isTainted);

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
  return secretTaintedNodes.count(uniqueNodeId) > 0 && secretTaintedNodes.at(uniqueNodeId);
}

const SecretTaintedNodesMap &SpecialTypeCheckingVisitor::getSecretTaintedNodes() const {
  return secretTaintedNodes;
}

SecretTaintedNodesMap &SpecialTypeCheckingVisitor::getSecretTaintedNodes() {
  return secretTaintedNodes;
}

void SpecialTypeCheckingVisitor::addVariableDatatype(ScopedIdentifier &scopedIdentifier, Datatype datatype) {
  variablesDatatypeMap.insert_or_assign(scopedIdentifier, datatype);
}

void SpecialTypeCheckingVisitor::registerBeforeVisitChildren() {
  numNodesBeforeVisitingChildren = typesVisitedNodes.size();
}

void SpecialTypeCheckingVisitor::discardChildrenDatatypes() {
  if (numNodesBeforeVisitingChildren > typesVisitedNodes.size()) {
    throw std::runtime_error(
        "Cannot discard datatype of children as numNodesBeforeVisitingChildren > number of current "
        "nodes in typesVisitedNodes. Did you forget calling registerBeforeVisitChildren() before "
        "visiting the children?");
  } else {
    while (typesVisitedNodes.size() > numNodesBeforeVisitingChildren) {
      typesVisitedNodes.pop();
    }
  }
}
