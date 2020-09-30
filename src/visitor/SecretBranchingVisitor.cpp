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

void SpecialSecretBranchingVisitor::addIdentifiers(Scope &scope) {
  std::for_each(expressionValues.begin(), expressionValues.end(),
                [&](std::pair<ScopedIdentifier, AbstractExpression *> key) {
                  scope.addIdentifier(key.first.getId());
                });
}

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
    std::unique_ptr<AbstractExpression> &&condition,
    std::unique_ptr<AbstractExpression> &&trueValue,
    std::unique_ptr<AbstractExpression> &&falseValue = nullptr) {
  auto conditionLhs = std::move(condition);
  auto conditionRhs = std::make_unique<BinaryExpression>(std::make_unique<LiteralInt>(1),
                                                         Operator(FHE_SUBTRACTION),
                                                         std::move(conditionLhs->clone(nullptr)));
  auto clauseLhs =
      std::make_unique<BinaryExpression>(std::move(conditionLhs), Operator(FHE_MULTIPLICATION), std::move(trueValue));

  if (falseValue!=nullptr) {
    auto clauseRhs =
        std::make_unique<BinaryExpression>(std::move(conditionRhs),
                                           Operator(FHE_MULTIPLICATION),
                                           std::move(falseValue));
    auto newExpr =
        std::make_unique<BinaryExpression>(std::move(clauseLhs), Operator(FHE_ADDITION), std::move(clauseRhs));
    // returns condition*trueValue + (1-c)*falseValue
    return std::make_unique<Assignment>(std::move(assignmentTarget), std::move(newExpr));
  } else {
    // returns condition*trueValue
    return std::make_unique<Assignment>(std::move(assignmentTarget), std::move(clauseLhs));
  }
}

void SpecialSecretBranchingVisitor::visit(If &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
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
    // TODO: Implement rewriting of If statement by considering:
    // - if branch only
    // - if + else branch
    // - uninitialized variables (declared only)

    visitedStatementMarkedForDeletion = true;

    // if branch exists only: rewrite to (c)*thenValue+(1-c)*oldValue
    if (!node.hasElseBranch()) {

      // get those values that were actually changed in the Then branch
      auto changedVars = getChangedVariables(exprValuesBefore, exprValuesAfterThen);

      for (auto &[scopedIdentifer, abstractExp] : changedVars) {
        std::unique_ptr<Assignment> assignm;
        if (exprValuesBefore.count(scopedIdentifer)!=0) {
          assignm = createDependentAssignment(
              std::make_unique<Variable>(scopedIdentifer.getId()),
              std::move(node.getCondition().clone(nullptr)),
              std::move(abstractExp->clone(nullptr)),
              std::move(exprValuesBefore.at(scopedIdentifer)->clone(nullptr)));
        } else {
          assignm = createDependentAssignment(
              std::make_unique<Variable>(scopedIdentifer.getId()),
              std::move(node.getCondition().clone(nullptr)),
              std::move(abstractExp->clone(nullptr)));
        }
        replacementStatements.push_back(std::move(assignm));
      }

      std::cout << "else" << std::endl;


//      std::vector<std::unique_ptr<AbstractStatement>> statements;
//      for (auto &scopedIdentifier : exprValuesAfterThen) {
//        // a variable with an existing value before the Then branch -> get oldValue
//        if (exprValuesBefore.count(scopedIdentifier.first) > 0 && ) {
//
//          auto assignment = createDependentAssignment(
//              std::make_unique<Variable>(scopedIdentifier.first.getId()),
//              std::unique_ptr<AbstractExpression>(node.getCondition().clone(nullptr)),
//              std::unique_ptr<AbstractExpression>(scopedIdentifier.second->clone(nullptr)),
//              std::unique_ptr<AbstractExpression>(exprValuesBefore.at(scopedIdentifier.first)->clone(nullptr)));
//          statements.push_back(std::move(assignment));
//        }
//      }
    }
  }
}

void SpecialSecretBranchingVisitor::visit(For &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Block &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
  decltype(node.getStatementPointers().begin()) insertionPos;
  auto it = node.getStatementPointers().begin();
  while (it!=node.getStatementPointers().end()) {
//    auto &ref = *it;
    it->get()->accept(*this);
    if (visitedStatementMarkedForDeletion) {
      visitedStatementMarkedForDeletion = false;
      (*it).reset();
      insertionPos = it;
    }
    it++;
  }

  it = node.getStatementPointers().insert(insertionPos,
                                          std::make_move_iterator(replacementStatements.begin()),
                                          std::make_move_iterator(replacementStatements.end()));
  replacementStatements.clear();

  node.removeNullStatements();
}

void SpecialSecretBranchingVisitor::visit(Return &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Assignment &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
  if (auto lhsVariable = dynamic_cast<Variable *>(&node.getTarget())) {
    auto scopedIdentifier = getCurrentScope().resolveIdentifier(lhsVariable->getIdentifier());
    expressionValues.insert_or_assign(scopedIdentifier, &node.getValue());
  } else {
    throw std::runtime_error("SecretBranchingVisitor can handle assignments to variables yet only.");
  }
}

void SpecialSecretBranchingVisitor::visit(VariableDeclaration &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
  AbstractExpression *value = (node.hasValue()) ? &node.getValue() : nullptr;
  expressionValues.insert_or_assign(ScopedIdentifier(getCurrentScope(), node.getTarget().getIdentifier()), value);
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(FunctionParameter &node) {
  std::cout << "Visiting " << node.getUniqueNodeId() << std::endl;
  // for FunctionParameters, we never know their actual value at this point
  expressionValues.insert_or_assign(ScopedIdentifier(getCurrentScope(), node.getIdentifier()), nullptr);
  ScopedVisitor::visit(node);
}

SpecialSecretBranchingVisitor::SpecialSecretBranchingVisitor(SecretTaintedNodesMap &taintedNodesMap)
    : secretTaintedNodesMap(taintedNodesMap) {
}
