#include <queue>
#include <iostream>

#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/ExpressionBatcher.h"
#include "ast_opt/visitor/Vectorizer.h"

void SpecialVectorizer::visit(Block &elem) {
  ScopedVisitor::enterScope(elem);
  variableValues.resetChangeFlags();

  for (auto &p: elem.getStatementPointers()) {
    p->accept(*this);
    if (delete_flag) { p.reset(); }
    delete_flag = false;
  }
  elem.removeNullStatements();

  // TODO: Emit all relevant assignments again!
  for (auto &scopedID: variableValues.changedEntries()) {
    auto &cv = variableValues.erase(scopedID);
//    for (auto &statement : cv.statementsToExecutePlan()) {
//      elem.appendStatement(std::move(statement));
//    }
  }
  variableValues.resetChangeFlags();
  ScopedVisitor::exitScope();
}

std::pair<ScopedIdentifier,ComplexValue> SpecialVectorizer::turnAssignmentIntoComplexValue(Assignment &elem) {
  if (auto indexAccessPtr = dynamic_cast<IndexAccess *>(&elem.getTarget())) {
    if (auto targetVariablePtr = dynamic_cast<Variable *>(&indexAccessPtr->getTarget())) {
      if (auto literalIntPtr = dynamic_cast<LiteralInt *>(&indexAccessPtr->getIndex())) {
        ScopedIdentifier scopedId = getCurrentScope().resolveIdentifier(targetVariablePtr->getIdentifier());
        return {scopedId, ComplexValue(elem.getValue(), literalIntPtr->getValue())};
      } else {
        throw std::runtime_error("Assignments to an IndexAccess must have a LiteralInt target slot.");
      }
    } else {
      throw std::runtime_error("Assignments to nested index accesses are currently not supported.");
    }
  } else if (auto targetVariablePtr = dynamic_cast<Variable *>(&elem.getTarget())) {
    ScopedIdentifier scopedId = getCurrentScope().resolveIdentifier(targetVariablePtr->getIdentifier());
    return {scopedId, ComplexValue(elem.getValue(), -1)};
  } else {
    throw std::runtime_error("Unexpected type of target in assignment.");
  }
}

void SpecialVectorizer::visit(Assignment &elem) {

  /// current scope
  auto &scope = getCurrentScope();

  /// target of the assignment
  AbstractTarget &target = elem.getTarget();
  ScopedIdentifier targetID;
  ComplexValue targetValue;


   /// TODO: Optimize the value of the assignment
  std::tie(targetID,targetValue) = turnAssignmentIntoComplexValue(elem);

  /// Combine the execution plans, if they already exist
  if (variableValues.has(targetID)) {
    ComplexValue &value = variableValues.get(targetID);
    value.merge(targetValue); //Since this only changes the object, the reference in the variableValues should still be correct.
  } else {
    precomputedValues.push_back(targetValue);
    variableValues.add(targetID, precomputedValues[precomputedValues.size() - 1]);
  }

  // Now delete this assignment
  delete_flag = true;
}

std::string SpecialVectorizer::getAuxiliaryInformation() {
  //TODO: Implement returning of auxiliary information
  return "NOT IMPLEMENTED YET";
}

bool isTransparentNode(const AbstractNode &node) {
  // a node is considered as transparent if it is an OperatorExpr because it can be batched by expanding any other
  // expression using the neutral element e.g., a and b*2 –- can be batched as a*1 and b*2
  return dynamic_cast<const BinaryExpression *>(&node)!=nullptr
      || dynamic_cast<const UnaryExpression *>(&node)!=nullptr
      || dynamic_cast<const OperatorExpression *>(&node)!=nullptr;
}

bool isBatchingCompatible(AbstractNode &baseNode, AbstractNode &curNode) {
  if (typeid(baseNode)!=typeid(curNode)) {
    // return true if...
    // - exactly one of both is transparent:
    //   (A XOR B)
    //   <=> (A && !B) || (!A && B)
    //   <=> (!A != !B)
    // - one of both is a AbstractLiteral
    return (!isTransparentNode(baseNode)!=!isTransparentNode(curNode))
        || isLiteral(baseNode)
        || isLiteral(curNode);
  } else {  // baseNode.type == curNode.type
    // type-specific checks
    if (auto baseNodeAsMatrixElementRef = dynamic_cast<IndexAccess *>(&baseNode)) {
      auto baseNodeVar = dynamic_cast<Variable *>(&baseNodeAsMatrixElementRef->getTarget());
      // as baseNode's type equals curNode's type, we know that curNodeAsMatrixElementRef != nullptr
      auto curNodeAsMatrixElementRef = dynamic_cast<IndexAccess *>(&curNode);
      auto curNodeVar = dynamic_cast<Variable *>(&curNodeAsMatrixElementRef->getTarget());
      if (baseNodeVar==nullptr || curNodeVar==nullptr) {
        throw std::runtime_error("IndexAccess unexpectedly does not refer to a Variable");
      }
      // check that both MatrixElementRefs refer to the same variable
      return baseNodeVar->getIdentifier()==curNodeVar->getIdentifier();
    } else if (auto baseNodeAsOperatorExpr = dynamic_cast<OperatorExpression *>(&baseNode)) {
      auto curNodeAsOperatorExpr = dynamic_cast<OperatorExpression *>(&curNode);
      // same operator
      return baseNodeAsOperatorExpr->getOperator()==curNodeAsOperatorExpr->getOperator()
          // same number of operands
          && baseNodeAsOperatorExpr->getOperands().size()==curNodeAsOperatorExpr->getOperands().size();
    } else { //TODO: Handle BinaryExpression and UnaryExpression!
      // handles all types that do not require any special handling, e.g., LiteralInt, Variable
      // (it is sufficient for batching compatibility that baseNode and curNode have the same type in that case)
      return true;
    }
  }
}