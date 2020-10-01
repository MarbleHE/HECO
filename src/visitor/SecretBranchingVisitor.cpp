#include "ast_opt/visitor/SecretBranchingVisitor.h"

#include <iostream>

#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/utilities/Operator.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Block.h"

VariableValueMap SpecialSecretBranchingVisitor::getChangedVariables(const VariableValueMap &baseMap,
                                                                    const VariableValueMap &changedMap) {
  VariableValueMap result;
  for (auto &[k, v] : changedMap) {
    // a newly declared variable or a variable whose value changed
    if ((baseMap.count(k)==0) || (v!=baseMap.at(k))) {
      result.insert_or_assign(k, v);
    }
  }
  return result;
}

std::unique_ptr<Assignment> createDependentAssignment(
    std::unique_ptr<AbstractTarget> assignmentTarget,
    std::unique_ptr<AbstractExpression> &&branchingCondition,
    std::unique_ptr<AbstractExpression> &&trueValue,
    std::unique_ptr<AbstractExpression> &&falseValue) {

  auto conditionThen = std::move(branchingCondition);
  auto conditionElse = std::make_unique<BinaryExpression>(std::make_unique<LiteralInt>(1),
                                                          Operator(FHE_SUBTRACTION),
                                                          std::move(conditionThen->clone(nullptr)));
  if (trueValue!=nullptr) {
    auto clauseTrue = std::make_unique<BinaryExpression>(std::move(conditionThen),
                                                         Operator(FHE_MULTIPLICATION),
                                                         std::move(trueValue));
    if (falseValue!=nullptr) {  // rewrite to: (c)*trueValue + (1-c)*falseValue
      auto clauseRhs = std::make_unique<BinaryExpression>(std::move(conditionElse),
                                                          Operator(FHE_MULTIPLICATION),
                                                          std::move(falseValue));
      auto newExpr = std::make_unique<BinaryExpression>(std::move(clauseTrue),
                                                        Operator(FHE_ADDITION),
                                                        std::move(clauseRhs));
      return std::make_unique<Assignment>(std::move(assignmentTarget), std::move(newExpr));
    } else {  // rewrite to: (c)*trueValue
      return std::make_unique<Assignment>(std::move(assignmentTarget), std::move(clauseTrue));
    }
  } else if (falseValue!=nullptr) {  // rewrite to: (1-c)*falseValue
    auto clauseFalse = std::make_unique<BinaryExpression>(std::move(conditionElse),
                                                          Operator(FHE_MULTIPLICATION),
                                                          std::move(falseValue));
    return std::make_unique<Assignment>(std::move(assignmentTarget), std::move(clauseFalse));
  } else {
    throw std::runtime_error("Cannot create dependent assignment where both trueValue and falseValue are not given!");
  }
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

  // exit if there is no secret condition involved or we visited an unsupported body statement in the If statement's
  // Then or Else branch
  if (!isSecretCondition || unsupportedBodyStatementVisited)
    return;

  // == perform the secret branching removal ===============

  visitedStatementMarkedForDeletion = true;

  // if Then branch exists only: rewrite to (c)*thenValue+(1-c)*oldValue
  if (!node.hasElseBranch()) {
    // get those values that were actually changed in the Then branch
    auto changedVars = getChangedVariables(exprValuesBefore, exprValuesAfterThen);
    // process each variable modified in the Then branch
    for (auto &[scopedIdentifer, abstractExp] : changedVars) {
      std::unique_ptr<Assignment> assignm;
      // check if the variable had a value before entering the Then branch, or if it is a new variable declaration
      if (exprValuesBefore.count(scopedIdentifer)!=0) {
        // oldValue is either a concrete AbstractExpression or, if the variable was declared but not initialized, we
        // simply assign the variable to itself (e.g., sum = c*trueValue + (1-c)*sum)
        std::unique_ptr<AbstractExpression> oldValue =
            (exprValuesBefore.at(scopedIdentifer)==nullptr)
            ? std::make_unique<Variable>(scopedIdentifer.getId())
            : std::unique_ptr<AbstractExpression>(exprValuesBefore.at(scopedIdentifer)->clone(nullptr));
        assignm = createDependentAssignment(
            std::make_unique<Variable>(scopedIdentifer.getId()),
            std::move(node.getCondition().clone(nullptr)),
            std::move(abstractExp->clone(nullptr)),
            std::move(oldValue));
      } else {
        // case where variable was declared in the Then branch only
        assignm = createDependentAssignment(
            std::make_unique<Variable>(scopedIdentifer.getId()),
            std::move(node.getCondition().clone(nullptr)),
            std::move(abstractExp->clone(nullptr)),
            nullptr);
      }
      expressionValues.insert_or_assign(scopedIdentifer, &assignm->getValue());
      replacementStatements.push_back(std::move(assignm));
    }
  } else { // node has Then and Else branch
    // get the variables (and their values) that were changed in the Then or Else branch
    auto thenModified = getChangedVariables(exprValuesBefore, exprValuesAfterThen);
    auto elseModified = getChangedVariables(exprValuesBefore, exprValuesAfterElse);
    // process each variable modified in the Then branch
    for (auto &[scopedIdentifer, abstractExp] : thenModified) {
      std::unique_ptr<Assignment> assignm;
      // check if the variable had a value before entering the Then branch, or if it is a new variable declaration
      if (elseModified.count(scopedIdentifer)!=0) {
        assignm = createDependentAssignment(
            std::make_unique<Variable>(scopedIdentifer.getId()),
            std::move(node.getCondition().clone(nullptr)),
            std::move(abstractExp->clone(nullptr)),
            std::move(elseModified.at(scopedIdentifer)->clone(nullptr)));
        // this time we need to delete the variables from the elseModified map to remember the ones that were only
        // changed in the Else branch
        elseModified.erase(scopedIdentifer);
      } else {
        // case where variable is not changed in the Else branch
        assignm = createDependentAssignment(
            std::make_unique<Variable>(scopedIdentifer.getId()),
            std::move(node.getCondition().clone(nullptr)),
            std::move(abstractExp->clone(nullptr)),
            nullptr);
      }
      expressionValues.insert_or_assign(scopedIdentifer, &assignm->getValue());
      replacementStatements.push_back(std::move(assignm));
    }
    // now go through the list of all variables that were only changed in the Else branch
    for (auto &[scopedIdentifer, abstractExp] : elseModified) {
      std::unique_ptr<Assignment> assignm = createDependentAssignment(
          std::make_unique<Variable>(scopedIdentifer.getId()),
          std::move(node.getCondition().clone(nullptr)),
          nullptr,
          std::move(abstractExp->clone(nullptr)));
      expressionValues.insert_or_assign(scopedIdentifer, &assignm->getValue());
      replacementStatements.push_back(std::move(assignm));
    }
  }
}

void SpecialSecretBranchingVisitor::visit(For &node) {
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Block &node) {
  decltype(node.getStatementPointers().end()) insertionPos;
  auto it = node.getStatementPointers().begin();
  // iterate over all children of this Block
  while (it!=node.getStatementPointers().end()) {
    it->get()->accept(*this);
    // if the recently visited statement set the visitedStatementMarkedForDeletion flag, then reset it's unique_ptr to
    // nullptr (these will be removed after looping over the elements) and remember the iterator's position
    if (visitedStatementMarkedForDeletion) {
      visitedStatementMarkedForDeletion = false;
      (*it).reset();
      insertionPos = it;
    }
    it++;
  }

  // insert the nodes in replacementStatements in the position indicated by insertionPos, i.e., the position where the
  // node marked for deletion was before
  if (insertionPos!=node.getStatementPointers().end()) {
    it = node.getStatementPointers().insert(insertionPos,
                                            std::make_move_iterator(replacementStatements.begin()),
                                            std::make_move_iterator(replacementStatements.end()));
    replacementStatements.clear();
  }

  // remove all unique_ptr that are null from the Block's list of statements
  node.removeNullStatements();
}

void SpecialSecretBranchingVisitor::visit(Return &node) {
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Assignment &node) {
  // this visitor only considers assignments to plain variables as we cannot rewrite dependent assignments involving
  // array accesses yet
  if (auto lhsVariable = dynamic_cast<Variable *>(&node.getTarget())) {
    auto scopedIdentifier = getCurrentScope().resolveIdentifier(lhsVariable->getIdentifier());
    expressionValues.insert_or_assign(scopedIdentifier, &node.getValue());
  }
}

void SpecialSecretBranchingVisitor::visit(VariableDeclaration &node) {
  // if the variable is declared + initialized, save variable identifier + value, otherwise use "nullptr" as value
  AbstractExpression *value = (node.hasValue()) ? &node.getValue() : nullptr;
  expressionValues.insert_or_assign(ScopedIdentifier(getCurrentScope(), node.getTarget().getIdentifier()), value);
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(FunctionParameter &node) {
  // for FunctionParameters, we never know their actual value at this point thus we set their value to "nullptr"
  expressionValues.insert_or_assign(ScopedIdentifier(getCurrentScope(), node.getIdentifier()), nullptr);
  ScopedVisitor::visit(node);
}

SpecialSecretBranchingVisitor::SpecialSecretBranchingVisitor(SecretTaintedNodesMap &taintedNodesMap)
    : secretTaintedNodesMap(taintedNodesMap) {
}
