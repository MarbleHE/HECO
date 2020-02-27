#include <unordered_set>
#include <DotPrinter.h>
#include <NodeUtils.h>
#include "CompileTimeExpressionSimplifier.h"
#include "ArithmeticExpr.h"
#include "LogicalExpr.h"
#include "LiteralFloat.h"
#include "VarDecl.h"
#include "Variable.h"
#include "VarAssignm.h"
#include "UnaryExpr.h"
#include "Block.h"
#include "Return.h"
#include "If.h"
#include "Function.h"
#include "FunctionParameter.h"
#include "ParameterList.h"

CompileTimeExpressionSimplifier::CompileTimeExpressionSimplifier() {
  evalVisitor = EvaluationVisitor();
}

// =====================
// AST objects that do not require or allow any simplifications
// =====================

void CompileTimeExpressionSimplifier::visit(AbstractNode &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(AbstractExpr &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(AbstractStatement &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(Operator &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(Ast &elem) {
  Visitor::visit(elem);
  // Delete all noted queued for deletion after finishing the simplification traversal.
  // It's important that we perform deletion in a FIFO-style because otherwise it can happen that we delete an enclosing
  // statement after trying to delete its child that is still in nodesQueuedForDeletion. However, the child is already
  // non-existent as we perform deletion recursively on the enclosing statement including the whole subtree.
  std::set<AbstractNode *> nodesAlreadyDeleted;
  while (!nodesQueuedForDeletion.empty()) {
    auto nodeToBeDeleted = nodesQueuedForDeletion.front();
    nodesQueuedForDeletion.pop();
    if (nodesAlreadyDeleted.count(nodeToBeDeleted) > 0) continue;
    nodesAlreadyDeleted.insert(nodeToBeDeleted);
    elem.deleteNode(&nodeToBeDeleted, true);
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getRootNode());
}

void CompileTimeExpressionSimplifier::visit(Datatype &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(CallExternal &elem) {
  Visitor::visit(elem);
}

// =====================
// AST objects that are trivially evaluable (AST leaf nodes)
// =====================

/** @defgroup visit leaves Methods for handling visit(...) of leaf nodes.
 *    These methods add a clone of the node (with its value) as evaluation result.
 *    It's important that a clone is added here because a variable declaration (VarDecl) such as "bool b = true;" will
 *    be removed and b's value will be copied as a variable-value pair (b, *LiteralBool(true)) into the map
 *    variableValues. After finishing the simplification traversal, the node VarDecl containing "bool b = true;" and all
 *    of its children, including LiteralBool(true), will be deleted. Hence, looking up any value in variableValues after
 *    the traversal is not possible anymore. However, as this is needed by tests we store a clone here.
 *    Note: storeEvaluatedNode creates and adds such a cloned node.
 *  @{
 */

void CompileTimeExpressionSimplifier::visit(LiteralBool &elem) {
  storeEvaluatedNode(&elem, &elem);
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralInt &elem) {
  storeEvaluatedNode(&elem, &elem);
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralString &elem) {
  storeEvaluatedNode(&elem, &elem);
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralFloat &elem) {
  storeEvaluatedNode(&elem, &elem);
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(Variable &elem) {
  Visitor::visit(elem);
  if (valueIsKnown(&elem)) {
    // if the variable's value is known, mark the node as evaluated
    storeEvaluatedNode(&elem, getFirstValue(&elem));
  } else if (variableValues.count(elem.getIdentifier()) > 0) {
    // if we know the variable's symbolic value instead, we can replace this variable node by the AbstractExpr that
    // defines the variable's value
    auto variableParent = elem.getParentsNonNull().front();
    elem.isolateNode();
    nodesQueuedForDeletion.push(&elem);
    variableParent->addChild(
        variableValues.at(elem.getIdentifier())->clone(false)->castTo<AbstractExpr>(),
        true);
  }
}

/** @} */ // End of visit leaves group


// =====================
// Simplifiable Statements
// =====================

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  Visitor::visit(elem);

  // stop execution here if there is an initializer but the initializer's value is not known
  if (elem.getInitializer()!=nullptr && !valueIsKnown(elem.getInitializer())) {
    // clean up temporary results from children in evaluatedNodes map
    evaluatedNodes.erase(elem.getInitializer());
    return;
  }

  AbstractExpr *variableValue;
  if (elem.getInitializer()==nullptr) {
    // If this variable was declared but not initialized, then we store the variable's default value (e.g., 0 for int,
    // 0.0f for float) in the map. This is required because if this variable will be changed in an If statement, we need
    // to have the variable's original value for rewriting the If statement.
    variableValue = getDefaultVariableInitializationValue(elem.getDatatype()->getType());
  } else if (valueIsKnown(elem.getInitializer())) {
    // If the initializer's value is known, we assign its value to the variable this VarDecl refers to.
    variableValue = getFirstValue(elem.getInitializer());
  }
  // store the variable's value
  variableValues.emplace(elem.getVarTargetIdentifier(), variableValue);

  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getInitializer());

  // mark this statement for deletion as we don't need it anymore (we don't care about the lost type information)
  nodesQueuedForDeletion.push(&elem);

  // mark this statement as evaluated to notify its parent that this can be considered as evaluated
  storeEvaluatedNode(&elem, variableValue);
}

void CompileTimeExpressionSimplifier::visit(VarAssignm &elem) {
  Visitor::visit(elem);

  AbstractExpr *nodeValue = nullptr;
  if (valueIsKnown(elem.getValue())) {
    // if the value is known, we'll assign the evaluated value to the variable
    nodeValue = getFirstValue(elem.getValue());
    // mark this statement as evaluated to notify its parent about a processed child and store the variable's result
    storeEvaluatedNode(&elem, nodeValue);
  } else {
    // otherwise, we use the expression itself but do not mark the node as evaluated
    nodeValue = elem.getValue();
  }
  variableValues[elem.getVarTargetIdentifier()] = nodeValue->clone(false)->castTo<AbstractExpr>();
  // mark this statement for deletion as we don't need it anymore
  nodesQueuedForDeletion.push(&elem);
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getValue());
}

void CompileTimeExpressionSimplifier::handleBinaryExpressions(AbstractNode &arithmeticExpr,
                                                              AbstractExpr *leftOperand,
                                                              AbstractExpr *rightOperand) {
  auto leftValueIsKnown = valueIsKnown(leftOperand);
  auto rightValueIsKnown = valueIsKnown(rightOperand);
  if (leftValueIsKnown && rightValueIsKnown) {
    // if both operand values are known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&arithmeticExpr, getTransformedVariableMap());
    storeEvaluatedNode(&arithmeticExpr, std::vector<AbstractExpr *>(result.begin(), result.end()));
  } else if (leftValueIsKnown || rightValueIsKnown) {
    // if only one of both is known -> simplify expression by replacing operand's value by the evaluation result
    auto concernedOperand = leftValueIsKnown ? leftOperand : rightOperand;
    auto newValue = getFirstValue(concernedOperand);
    arithmeticExpr.removeChildBilateral(concernedOperand);
    arithmeticExpr.addChild(newValue);
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(leftOperand);
  evaluatedNodes.erase(rightOperand);
  evaluatedNodes.erase(arithmeticExpr.getChildAtIndex(1)); // operator
}

void CompileTimeExpressionSimplifier::simplifyNestedBinaryExpressions(AbstractExpr *nestedExprRoot) {
  // check if this is a nested binary expression at all
  bool containsNestedExpression = false;
  for (auto &expr : nestedExprRoot->getChildrenNonNull()) {
    if (dynamic_cast<ArithmeticExpr *>(expr) || dynamic_cast<LogicalExpr *>(expr)) {
      containsNestedExpression = true;
      break;
    }
  }
  // stop doing anything in this method if this is not a nested binary expression
  if (!containsNestedExpression) return;

  // create a function that determines the operator of a binary expression (as we'll need it later again)
  auto getOperatorSymbolOfBinaryExpression = [](AbstractNode *node) {
    if (auto rootAsArithmeticExpr = dynamic_cast<ArithmeticExpr *>(node))
      return rootAsArithmeticExpr->getOp()->getOperatorSymbol();
    else if (auto rootAsLogicalExpr = dynamic_cast<LogicalExpr *>(node))
      return rootAsLogicalExpr->getOp()->getOperatorSymbol();
    else
      throw std::logic_error("getOperatorSymbolOfBinaryExpression expects an Arithmetic or Logical Expression!");
  };
  // memorize the current operator
  std::variant<OpSymb::ArithmeticOp, OpSymb::LogCompOp, OpSymb::UnaryOp> currentOperator =
      getOperatorSymbolOfBinaryExpression(nestedExprRoot);

  // enqueue the starting node
  std::queue<AbstractNode *> nodesInBinaryExpressionsWithSameOperator;
  std::queue<AbstractNode *> nodesInBinaryExpressionsWithOtherOperator({nestedExprRoot});

  // create a vector for all literals and all symbols
  std::vector<AbstractExpr *> literalsWithSameOperator;
  std::vector<AbstractExpr *> variablesWithSameOperator;

  // TODO only consider commutative operators:
  //  addition, subtraction, multiplication, AND, OR, XOR

  // TODO consider the case where literalsWithSameOperator has only 0 or 1 element -> not simplifiable

  while (!nodesInBinaryExpressionsWithOtherOperator.empty()) {
    auto firstNodeOfOperatorToBeProcessedNext = nodesInBinaryExpressionsWithOtherOperator.front();

    // clean up
    currentOperator = getOperatorSymbolOfBinaryExpression(firstNodeOfOperatorToBeProcessedNext);
    literalsWithSameOperator.clear();
    variablesWithSameOperator.clear();

    nodesInBinaryExpressionsWithOtherOperator.pop();
    nodesInBinaryExpressionsWithSameOperator.push(firstNodeOfOperatorToBeProcessedNext);

    // traverse the whole tree
    while (!nodesInBinaryExpressionsWithSameOperator.empty()) {
      auto curNode = nodesInBinaryExpressionsWithSameOperator.front();
      nodesInBinaryExpressionsWithSameOperator.pop();
      // check the type of all of the current node's children
      if (auto childAsLiteral = dynamic_cast<AbstractLiteral *>(curNode)) {  // child is a Literal -> memorize it
        literalsWithSameOperator.push_back(childAsLiteral);
      } else if (auto childAsVariable = dynamic_cast<Variable *>(curNode)) {  // child is a Variable -> memorize it
        variablesWithSameOperator.push_back(childAsVariable);
      } else if (dynamic_cast<ArithmeticExpr *>(curNode)
          || dynamic_cast<LogicalExpr *>(curNode)) {  // child is arithmetic or logical expr.
        // if this nested expression has the same operator -> continue
        if (getOperatorSymbolOfBinaryExpression(curNode)==currentOperator) {
          // enqueue all of the expression's children
          for (auto &childOfSubexpression : curNode->getChildrenNonNull())
            nodesInBinaryExpressionsWithSameOperator.push(childOfSubexpression);
        } else {  // otherwise simplify up to this part of the tree and continue with new operator
          // this is important because in an expression like (a + (7 + (12 * (4 + b)))) we don't want to mix up or
          // change the order of addition and multiplication
          nodesInBinaryExpressionsWithOtherOperator.push(curNode);
        }
      }

    } //end: nodesInBinaryExpressionsWithSameOperator

    // now we have processed all nodes of the current operator symbol
    // check if simplifying is possible at all, i.e., nested expressions contain more than two literals that can be
    // combined
    if (literalsWithSameOperator.size() >= 2) {
      // create subtree of literals only and evaluate it
      auto rewrittenSubtree =
          rewriteMultiInputBinaryExpressionToBinaryExpressionGatesChain(literalsWithSameOperator, currentOperator);
      auto result =
          evaluateNodeRecursive(rewrittenSubtree.back(),
                                std::unordered_map<std::string, AbstractLiteral *>()).front();

      // append the evaluation result (AbstractLiteral) to the map of variables
      variablesWithSameOperator.push_back(result);

      // create a chained binary expression with all variables and the literal
      auto exprChain =
          rewriteMultiInputBinaryExpressionToBinaryExpressionGatesChain(variablesWithSameOperator, currentOperator);

      if (nodesInBinaryExpressionsWithOtherOperator.empty())
        *nestedExprRoot = *exprChain.back()->castTo<AbstractExpr>();
      else
        throw std::logic_error("Not Implemented Exception: Merging of simplified subtrees missing yet!");

    }
  } //end: nodesInBinaryExpressionsWithOtherOperator
}

void CompileTimeExpressionSimplifier::visit(ArithmeticExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);
  handleBinaryExpressions(elem, elem.getLeft(), elem.getRight());
}

void CompileTimeExpressionSimplifier::visit(LogicalExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);
  handleBinaryExpressions(elem, elem.getLeft(), elem.getRight());
}

void CompileTimeExpressionSimplifier::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
  // check if the unary expression can be evaluated
  if (valueIsKnown(elem.getRight())) {
    // if operand value is known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&elem, getTransformedVariableMap());
    storeEvaluatedNode(&elem, std::vector<AbstractExpr *>(result.begin(), result.end()));
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getRight());
  evaluatedNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(Block &elem) {
  Visitor::visit(elem);
  // If all Block's statements are marked for deletion, mark it as evaluated to notify its parent.
  // The parent is then responsible to decide whether it makes sense to delete this Block or not.

  // check if there is any statement within this Block that is not marked for deletion
  bool allStatementsInBlockAreMarkedForDeletion = true;
  for (auto &statement : elem.getStatements()) {
    if (evaluatedNodes.count(statement)==0) {
      allStatementsInBlockAreMarkedForDeletion = false;
    }
    // clean up temporary results from children in evaluatedNodes map
    evaluatedNodes.erase(statement);
  }
  // if all statements of this Block are marked for deletion, we can mark the Block as evaluable
  // -> let the Block's parent decide whether to keep this empty Block or not
  if (allStatementsInBlockAreMarkedForDeletion) storeEvaluatedNode(&elem, nullptr);
}

void CompileTimeExpressionSimplifier::visit(Call &elem) {
  Visitor::visit(elem);
  // TODO(pjattke): implement me!
  //  We can replace the call node by the result of the inner function's return value.
  //  For example:
  //    int computeX(int input) {
  //      ...
  //      int z = computeZ(seed);   -->  int computeZ(int seed) { int v = genRandomNum(seed); return v + 32; }
  //      ...
  //    }
  //  is converted into:
  //    int computeX(int input) {
  //      ...
  //      int z = genRandomNum(seed) + 32;
  //      ...
  //    }
}

void CompileTimeExpressionSimplifier::visit(ParameterList &elem) {
  Visitor::visit(elem);
  // if all of the FunctionParameter children are marked as evaluable, mark this node as evaluable too
  auto parameters = elem.getParameters();
  bool allFunctionParametersAreEvaluable = true;
  for (auto &fp : elem.getParameters()) {
    if (!valueIsKnown(fp)) allFunctionParametersAreEvaluable = false;
    // clean up temporary results from children in evaluatedNodes map
    evaluatedNodes.erase(fp);
  }
  if (allFunctionParametersAreEvaluable) {
    storeEvaluatedNode(&elem, nullptr);
  }
}

void CompileTimeExpressionSimplifier::visit(Function &elem) {
  Visitor::visit(elem);
  // This node is marked as evaluable if both the ParameterList (implies all of the FunctionParameters) and the Block
  // (implies all of the statements) are evaluable. This mark is only relevant in case that this Function is referred
  // by a Call statement because Call statements can be replaced by inlining the Function's computation.
  if (valueIsKnown(elem.getParameterList()) && valueIsKnown(elem.getBody())) {
    storeEvaluatedNode(&elem, nullptr);
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getParameterList());
  evaluatedNodes.erase(elem.getBody());
}

void CompileTimeExpressionSimplifier::visit(FunctionParameter &elem) {
  Visitor::visit(elem);

  // This simplifier does not care about the variable's datatype, hence we can mark this node as evaluable (= deletion
  // candidate). This mark is only relevant in case that this FunctionParameter is part of a Function referred by a Call
  // statement because Call statements can be replaced by inlining the Function's computation.
  storeEvaluatedNode(&elem, nullptr);
}

void CompileTimeExpressionSimplifier::visit(If &elem) {
  // Bypass the base Visitor's logic and directly visit the condition only because we need to know whether it is
  // evaluable at runtime (or not) and its result.
  elem.getCondition()->accept(*this);

  // ================
  // Case 1: Condition's evaluation result is KNOWN at compile-time
  // -> we can delete the branch that is not executed and move or remove contained statements prior deleting the whole
  //    If statement including all of its children.
  // ================
  if (valueIsKnown(elem.getCondition())) {
    // If we know the condition's value, we can eliminate the branch that is never executed.
    // We need to detach this branch immediately, otherwise the variableValues map will only contain the values of the
    // last visited branch (but we need the values of the branch that is always executed instead).
    auto thenAlwaysExecuted = getFirstValue(elem.getCondition())->castTo<LiteralBool>()->getValue();
    if (thenAlwaysExecuted) { // the Then-branch is always executed
      // recursively remove the Else-branch (may not exist)
      if (elem.getElseBranch()!=nullptr) {
        nodesQueuedForDeletion.push(elem.getElseBranch());
        // we also unlink it from the If statement such that it will not be visited
        elem.removeChildBilateral(elem.getElseBranch());
      }
    } else {  // the Else-branch is always executed
      // recursively remove the Then-branch (always exists)
      nodesQueuedForDeletion.push(elem.getThenBranch());
      // negate the condition and delete the conditions stored value (is now invalid)
      auto condition = elem.getCondition();
      auto newCondition = new UnaryExpr(OpSymb::negation, condition);
      evaluatedNodes.erase(condition);
      // replace the If statement's Then branch by the Else branch
      elem.removeChildBilateral(elem.getThenBranch());
      auto elseBranch = elem.getElseBranch();
      elem.removeChildren();
      elem.setAttributes(newCondition, elseBranch, nullptr);
    }

    // continue visiting the remaining branches
    // (the condition will be visited again, but that's acceptable)
    Visitor::visit(elem);

    // check if we can remove the whole If statement
    bool wholeIfStmtCanBeDeleted = false;
    if (thenAlwaysExecuted) { // condition is always True => Then-branch is always executed
      if (valueIsKnown(elem.getThenBranch()))
        wholeIfStmtCanBeDeleted = true;
    } else { // condition is always False => Else-branch is always executed
      // The whole If statement can be deleted if there exists no Else branch (If body is not executed at all) or if the
      // Else-branch is evaluable
      if (elem.getElseBranch()==nullptr || valueIsKnown(elem.getElseBranch()))
        wholeIfStmtCanBeDeleted = true;
    }
    // enqueue the If statement and its children for deletion
    if (wholeIfStmtCanBeDeleted) nodesQueuedForDeletion.push(&elem);
  }
    // ================
    // Case 2: Condition's evaluation result is UNKNOWN at compile-time
    // -> rewrite variables values that are modified in either one or both of the If statement's branches such that the
    //    variable's value depends on the If statement's condition evaluation result.
    // ================
  else { // if we don't know the evaluation result of the If statement's condition -> rewrite the If statement
    // create a copy of the variableValues map and evaluatedNodes map
    std::unordered_map<std::string, AbstractExpr *> originalVariableValues(variableValues);

    // visit the thenBranch and store its modifications
    elem.getThenBranch()->accept(*this);
    std::unordered_map<std::string, AbstractExpr *> variableValuesAfterVisitingThen(variableValues);

    // check if there is an Else-branch that we need to visit
    if (elem.getElseBranch()!=nullptr) {
      // restore the original maps via copy assignments prior visiting Else-branch
      variableValues = originalVariableValues;

      // visit the Else-branch
      elem.getElseBranch()->accept(*this);
    }

    // rewrite those entries that were modified in either one or both maps
    // note: up to this point (and beyond), the Else-branch's modifications are in variableValues
    for (auto &[variableIdentifier, originalValue] : originalVariableValues) {
      // check if the variable was changed in the Then-branch
      auto thenBranchValue = variableValuesAfterVisitingThen.at(variableIdentifier);
      auto thenBranchModifiedCurrentVariable = (thenBranchValue!=originalValue);
      // check if the variable was changed in the Else-branch
      // if there is no Else-branch, elseBranchModifiedCurrentVariable stays False
      bool elseBranchModifiedCurrentVariable = false;
      AbstractExpr *elseBranchValue = nullptr;
      if (elem.getElseBranch()!=nullptr) {
        elseBranchValue = variableValues.at(variableIdentifier);
        elseBranchModifiedCurrentVariable = (elseBranchValue!=originalValue);
      }

      // determine if an If statement-dependent value needs to be assigned to the variable
      AbstractExpr *newValue;
      if (thenBranchModifiedCurrentVariable && elseBranchModifiedCurrentVariable) {
        newValue = generateIfDependentValue(elem.getCondition(), thenBranchValue, elseBranchValue);
      } else if (thenBranchModifiedCurrentVariable) {
        newValue = generateIfDependentValue(elem.getCondition(), thenBranchValue, originalValue);
      } else if (elseBranchModifiedCurrentVariable) {
        newValue = generateIfDependentValue(elem.getCondition(), originalValue, elseBranchValue);
      } else {
        // otherwise neither one of the two branches modified the variable's value and we can keep it unchanged
        continue;
      }
      // assign the new If statement-dependent value (e.g., myVarIdentifier = condition*32+[1-condition]*11)
      originalVariableValues[variableIdentifier] = newValue;
    }
    // restore the original map that contains the merged changes from the visited branches
    variableValues = originalVariableValues;

    // enqueue the If statement and its children for deletion
    nodesQueuedForDeletion.push(&elem);
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getCondition());
  evaluatedNodes.erase(elem.getThenBranch());
  evaluatedNodes.erase(elem.getElseBranch());
}

void CompileTimeExpressionSimplifier::visit(While &elem) {
  Visitor::visit(elem);
  // TODO(pjattke): implement me!
  // if bound of While-loop can be determined at compile-time -> unroll it by evaluating While statement
  // For example:
  //   ...
  //   int a = 12;
  //   int sum = 0;
  //   while (a > 0) {
  //     sum = sum - 2;
  //     a = a-1;
  //   }
  //   ...
  //  -- expected --
  //   ...
  //     sum = -24;
  //   ...
}

void CompileTimeExpressionSimplifier::visit(Return &elem) {
  Visitor::visit(elem);

  // If there was only one statement and this statement was evaluable, mark this Return node as evaluable.
  // If this Return is part of a Function that is nested into a Call and this Return is evaluable, then we can replace
  // the Call by the evaluated value. But only if there is a single return value because our AST cannot represent the
  // assignment of multiple return values yet.
  auto firstReturnExpression = elem.getReturnExpressions().front();
  if (elem.getReturnExpressions().size()==1 && valueIsKnown(firstReturnExpression)) {
    storeEvaluatedNode(&elem, getFirstValue(firstReturnExpression));
  }

  // simplify return expression: replace each evaluated expression by its evaluation result
  for (auto &returnExpr : elem.getReturnExpressions()) {
    // this simplification requires that the expression's value is known (i.e., was evaluated before)
    if (valueIsKnown(returnExpr)) {
      // remove the existing node from the return expression and add the simplified one
      auto newValue = getFirstValueOrExpression(returnExpr);
      elem.removeChildBilateral(returnExpr);
      elem.addChild(newValue);
      // mark this statement for deletion as we don't need it anymore
      nodesQueuedForDeletion.push(returnExpr);
    }
    simplifyNestedBinaryExpressions(returnExpr);
    // clean up temporary results from children in evaluatedNodes map
    evaluatedNodes.erase(returnExpr);
  }
}

// =====================
// Helper methods
// =====================

bool CompileTimeExpressionSimplifier::valueIsKnown(AbstractNode *abstractExpr) {
  // A value is considered as known if...
  // i.) it is a variable with a known value (in variableValues)
  auto variableValueIsKnown = false;
  if (auto abstractExprAsVariable = dynamic_cast<Variable *>(abstractExpr)) {
    // check that the variable has a value
    return variableValues.count(abstractExprAsVariable->getIdentifier()) > 0
        // and its value is not symbolic (i.e., contains a variable for which the value is not known)
        && variableValues.at(abstractExprAsVariable->getIdentifier())->getVariableIdentifiers().empty();
  }
  // ii.) or the node was evaluated before (i.e., its value is in evaluatedNodes)
  return variableValueIsKnown || evaluatedNodes.count(abstractExpr) > 0;
}

void CompileTimeExpressionSimplifier::storeEvaluatedNode(
    AbstractNode *node, const std::vector<AbstractExpr *> &evaluationResult) {
  std::vector<AbstractExpr *> evalResultCloned;
  evalResultCloned.reserve(evaluationResult.size());
  for (auto &exp : evaluationResult) {
    evalResultCloned.push_back(exp->clone(false)->castTo<AbstractExpr>());
  }
  evaluatedNodes.emplace(node, evalResultCloned);
}

void CompileTimeExpressionSimplifier::storeEvaluatedNode(
    AbstractNode *node, AbstractExpr *evaluationResult) {
  std::vector<AbstractExpr *> val;
  if (evaluationResult!=nullptr) {
    val.push_back(evaluationResult->clone(false)->castTo<AbstractExpr>());
  }
  evaluatedNodes.emplace(node, val);
}

std::vector<AbstractLiteral *> CompileTimeExpressionSimplifier::evaluateNodeRecursive(
    AbstractNode *n, std::unordered_map<std::string, AbstractLiteral *> valuesOfVariables) {
  // clean up the EvaluationVisitor from any previous run
  evalVisitor.reset();

  // perform evaluation by passing the required parameter values
  evalVisitor.updateVarValues(std::move(valuesOfVariables));
  n->accept(evalVisitor);

  // retrieve results
  return evalVisitor.getResults();
}

AbstractExpr *CompileTimeExpressionSimplifier::getFirstValue(AbstractNode *node) {
  // if node is a variable -> search the variable's value in the map of known variable values
  auto nodeAsVariable = dynamic_cast<Variable *>(node);
  if (nodeAsVariable!=nullptr && variableValues.count(nodeAsVariable->getIdentifier()) > 0) {
    return variableValues.at(nodeAsVariable->getIdentifier());
  }

  // in any other case -> search the given node in the map of already evaluated nodes
  auto it = evaluatedNodes.find(node);
  if (it==evaluatedNodes.end()) {
    // getFirstValue(...) assumes that the caller knows that the node was already evaluated (check using valueIsKnown
    // before calling it) -> throw an exception if no value was found.
    throw std::invalid_argument("Could not find any value for Node " + node->getUniqueNodeId() + ". ");
  } else if (it->second.size() > 1) {
    // if more than one evaluation results exists the user probably didn't know about that -> throw an exception
    throw std::logic_error("Using getFirstValue(...) for expressions returning more than one value is not supported.");
  } else {
    return it->second.front();
  }
}

AbstractExpr *CompileTimeExpressionSimplifier::getFirstValueOrExpression(AbstractNode *node) {
  // if there exists a AbstractLiteral value, then return this value (as AbstractExpr)
  auto firstValue = getFirstValue(node);
  if (firstValue!=nullptr) return firstValue;

  // if node is a variable -> search the variable's value in the map of known variable values
  // in addition to getFirstValue(...), we consider the AbstractExpr here
  auto nodeAsVariable = dynamic_cast<Variable *>(node);
  if (nodeAsVariable!=nullptr && variableValues.count(nodeAsVariable->getIdentifier()) > 0) {
    return variableValues.at(nodeAsVariable->getIdentifier());
  }
  return nullptr;
}

std::unordered_map<std::string, AbstractLiteral *> CompileTimeExpressionSimplifier::getTransformedVariableMap() {
  std::unordered_map<std::string, AbstractLiteral *> variableMap;
  for (auto &[k, v] : variableValues) {
    if (auto varAsLiteral = dynamic_cast<AbstractLiteral *>(v)) variableMap[k] = varAsLiteral;
  }
  return variableMap;
}

AbstractLiteral *CompileTimeExpressionSimplifier::getDefaultVariableInitializationValue(Types datatype) {
  switch (datatype) {
    case Types::BOOL:return new LiteralBool("false");
    case Types::INT:return new LiteralInt(0);
    case Types::FLOAT:return new LiteralFloat(0.0f);
    case Types::STRING:return new LiteralString("");
    default:
      throw std::invalid_argument("Unrecognized enum type given: Cannot determine its default value."
                                  "See enum Types for the supported types.");
  }
}
AbstractExpr *CompileTimeExpressionSimplifier::generateIfDependentValue(AbstractExpr *condition,
                                                                        AbstractExpr *trueValue,
                                                                        AbstractExpr *falseValue) {
  // We need to handle the case where trueValue or/and falseValue are null because in that case the dependent
  // statement can be simplified significantly by removing one/both operands of the arithmetic expression.

  // determine whether one or both of the provided expressions (trueValue, falseValue) are null
  auto exprIsNull = [](AbstractExpr *expr) {
    if (expr==nullptr) return true;
    auto castedExpr = dynamic_cast<AbstractLiteral *>(expr);
    return castedExpr!=nullptr && castedExpr->isNull();
  };
  auto trueValueIsNull = exprIsNull(trueValue);
  auto falseValueIsNull = exprIsNull(falseValue);

  // check whether both values are null
  if (trueValueIsNull && falseValueIsNull) {
    // case: trueValue == 0 && falseValue == 0 => expression always equals null, independent of condition's eval result
    // return a cloned copy of trueValue because we cannot directly create a new object (e.g., LiteralInt) as we do
    // not exactly know which subtype of AbstractLiteral trueValue has
    // return "0" (where 0 is of the respective input type)
    return trueValue->clone(false)->castTo<AbstractExpr>();
  }

  // check if exactly one of both values are null
  if (trueValueIsNull) {
    // factorIsFalse = [1-ifStatementCondition]
    auto factorIsFalse = new ArithmeticExpr(new LiteralInt(1),
                                            OpSymb::subtraction,
                                            condition->clone(false)->castTo<AbstractExpr>());
    // case: trueValue == 0 && falseValue != 0 => value is 0 if the condition is True -> return (1-b)*falseValue
    return new ArithmeticExpr(factorIsFalse, OpSymb::multiplication, falseValue);
  } else if (falseValueIsNull) {
    // factorIsTrue = ifStatementCondition
    auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
    // case: trueValue != 0 && falseValue == 0 => value is 0 if the condition is False -> return condition * trueValue
    return new ArithmeticExpr(factorIsTrue, OpSymb::multiplication, trueValue);
  }

  // default case: trueValue != 0 && falseValue != 0 => value is changed in both branches of If statement
  // -> return condition*trueValue + (1-b)*falseValue.
  // factorIsTrue = ifStatementCondition
  auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
  // factorIsFalse = [1-ifStatementCondition]
  auto factorIsFalse = new ArithmeticExpr(new LiteralInt(1),
                                          OpSymb::subtraction,
                                          condition->clone(false)->castTo<AbstractExpr>());
  return new ArithmeticExpr(
      new ArithmeticExpr(factorIsTrue,
                         OpSymb::multiplication,
                         trueValue->clone(false)->castTo<AbstractExpr>()),
      OpSymb::addition,
      new ArithmeticExpr(factorIsFalse,
                         OpSymb::multiplication,
                         falseValue->clone(false)->castTo<AbstractExpr>()));
}
