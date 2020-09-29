#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/visitor/SecretBranchingVisitor.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/If.h"
#include <iostream>

void SpecialSecretBranchingVisitor::addIdentifiers(Scope &scope) {
  std::for_each(expressionValues.begin(),
                expressionValues.end(),
                [&](std::pair<ScopedIdentifier, AbstractExpression *> key) {
                  scope.addIdentifier(key.first.getId());
                });
}

VariableValueMap SpecialSecretBranchingVisitor::getChangedVariables(const VariableValueMap &baseMap,
                                                                    const VariableValueMap &changedMap) {
  VariableValueMap result;

  for (auto &[k, v] : changedMap) {
    if ((baseMap.count(k)==0)     // a newly declared variable
        || (v!=changedMap.at(k))) { // a variable whose value changed
      result.insert_or_assign(k, v);
    }
  }

  return result;
}

void SpecialSecretBranchingVisitor::visit(If &node) {
  unsupportedBodyStatementVisited = false;

  VariableValueMap exprValuesBefore = expressionValues;
  VariableValueMap exprValuesAfterThen;
  VariableValueMap exprValuesAfterElse;

  // visit the if statement's condition
  node.getCondition().accept(*this);

  // visit the then branch
  node.getThenBranch().accept(*this);
  exprValuesAfterThen = expressionValues;

  // if existing, visit the else branch
  if (node.hasElseBranch()) {
    expressionValues = exprValuesBefore;
    node.getElseBranch().accept(*this);
    exprValuesAfterElse = expressionValues;
  }

  // check if the condition involves secret value
  auto conditionNodeId = node.getCondition().getUniqueNodeId();
  if (secretTaintedNodesMap.count(conditionNodeId)==0) {
    throw std::runtime_error(
        "Cannot find secret tainting information for condition node (" + conditionNodeId + ").");
  }
  auto isSecretCondition = secretTaintedNodesMap.at(conditionNodeId);

  if (isSecretCondition && !unsupportedBodyStatementVisited) {
    std::cout << "doing rewriting... " << std::endl;
//    if (!node.hasElseBranch()) {
//      for (auto &[k, v] :)
//      // rewrite to (c)*thenValue+(1-c)*oldValue
//
//
//    } else {
//      // rewrite to (c)*thenValue+(1-c)*elseValue
//    }
//
//
//    // get variables changed in both branches
//    if (sameChangedVariables(expressionValuesAfterThenBranch, expressionValuesAfterElseBranch)) {
//      // create statements
//    } else {
//
//    }

    // get variables changed in else branch only

    // get variables changed in both then and else branch


    // replace existing statement by new one

  }
}

void SpecialSecretBranchingVisitor::visit(For &node) {
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Return &node) {
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Assignment &node) {
  if (auto lhsVariable = dynamic_cast<Variable *>(&node.getTarget())) {
    auto scopedIdentifier = getCurrentScope().resolveIdentifier(lhsVariable->getIdentifier());
    expressionValues.insert_or_assign(scopedIdentifier, &node.getValue());
  } else {
    throw std::runtime_error("SecretBranchingVisitor can handle assignments to variables yet only.");
  }
}

void SpecialSecretBranchingVisitor::visit(VariableDeclaration &node) {
  AbstractExpression *value = nullptr;
  if (node.hasValue()) {
    value = &node.getValue();
  }
  expressionValues.insert_or_assign(
      ScopedIdentifier(getCurrentScope(), node.getTarget().getIdentifier()), value);
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(FunctionParameter &node) {
//  auto scopedIdentifier = std::make_unique<ScopedIdentifier>(getCurrentScope(), node.getIdentifier());
  expressionValues.insert_or_assign(ScopedIdentifier(getCurrentScope(), node.getIdentifier()), nullptr);
  ScopedVisitor::visit(node);
}

SpecialSecretBranchingVisitor::SpecialSecretBranchingVisitor(SecretTaintedNodesMap &taintedNodesMap)
    : secretTaintedNodesMap(taintedNodesMap) {

}
