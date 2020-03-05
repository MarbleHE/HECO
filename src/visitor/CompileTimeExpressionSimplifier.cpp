#include "CompileTimeExpressionSimplifier.h"
#include "DotPrinter.h"
#include "NodeUtils.h"
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
#include "CallExternal.h"
#include "While.h"

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
  // non-existent as we performed deletion recursively on the enclosing statement including its whole subtree.
  std::set<AbstractNode *> nodesAlreadyDeleted;
  while (!nodesQueuedForDeletion.empty()) {
    auto nodeToBeDeleted = nodesQueuedForDeletion.front();
    nodesQueuedForDeletion.pop_front();
    if (nodesAlreadyDeleted.count(nodeToBeDeleted) > 0) {
      throw std::runtime_error("ERROR: Trying to delete node twice. "
                               "Probably the node was by mistake enqueued multiple times for deletion.");
    }
    nodesAlreadyDeleted.insert(nodeToBeDeleted);
    elem.deleteNode(&nodeToBeDeleted, true);
  }
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getRootNode());
}

void CompileTimeExpressionSimplifier::visit(Datatype &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(CallExternal &elem) {
  Visitor::visit(elem);
  cleanUpAfterStatementVisited(elem.AbstractStatement::castTo<AbstractStatement>(), false);
}

// =====================
// AST objects that are leaf nodes and cannot be simplified
// =====================

void CompileTimeExpressionSimplifier::visit(LiteralBool &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralInt &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralString &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralFloat &elem) {
  Visitor::visit(elem);
}


// =====================
// Simplifiable Statements
// =====================

void CompileTimeExpressionSimplifier::visit(Variable &elem) {
  Visitor::visit(elem);
  if (variableValues.count(elem.getIdentifier()) > 0) {
    // if we know the variable's value (i.e., its value is either any subtype of AbstractLiteral or an AbstractExpr if
    // this is a symbolic value that defines on other variables), we can replace this variable node by its value
    auto variableParent = elem.getOnlyParent();
    auto newValue = variableValues.at(elem.getIdentifier())->clone(false)->castTo<AbstractExpr>();
    variableParent->replaceChild(&elem, newValue);
    // if this newValue is an AbstractBinaryExpr by itself, enqueue its operands
    if (auto bexp = dynamic_cast<AbstractBinaryExpr *>(newValue)) {
      binaryExpressionAccumulator.addOperands({bexp->getLeft(), bexp->getRight()});
      binaryExpressionAccumulator.setLastVisitedSubtree(bexp);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  Visitor::visit(elem);

  // determine the variable's value
  AbstractExpr *variableValue;
  if (elem.getInitializer()==nullptr) {
    variableValue = Datatype::getDefaultVariableInitializationValue(elem.getDatatype()->getType());
  } else {
    variableValue = elem.getInitializer();
  }
  // store the variable's value
  addVariableValue(elem.getVarTargetIdentifier(), variableValue);
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getInitializer());
  // mark this statement as removable as it is deleted
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::visit(VarAssignm &elem) {
  Visitor::visit(elem);
  // store the variable's value
  addVariableValue(elem.getVarTargetIdentifier(), elem.getValue());
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getValue());
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::handleBinaryExpression(AbstractBinaryExpr &arithmeticExpr) {
  // left-hand side operand
  auto lhsOperand = arithmeticExpr.getLeft();
  auto leftValueIsKnown = valueIsKnown(lhsOperand);
  // right-hand side operand
  auto rhsOperand = arithmeticExpr.getRight();
  auto rightValueIsKnown = valueIsKnown(rhsOperand);

  if (leftValueIsKnown && rightValueIsKnown) {
    // if both operand values are known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&arithmeticExpr, getTransformedVariableMap());
    if (result.size() > 1)
      throw std::logic_error("Unexpected: Evaluation result contains more than one value.");
    arithmeticExpr.getOnlyParent()->replaceChild(&arithmeticExpr, result.front());
    nodesQueuedForDeletion.push_back(&arithmeticExpr);
  } else {
    // update accumulator
    auto currentOperator = arithmeticExpr.getOp();

    // check if we need to reset the binaryExpressionAccumulator:
    // this is the case if either we are visiting the first binary expression in this statement or the operator of the
    // accumulated binary expressions does not match the one of this binary expression (we cannot accumulate more)
    auto isFirstBinaryExprInSubtree = !binaryExpressionAccumulator.containsOperands();
    auto operatorChangedSinceLastAccumulation =
        !currentOperator->equals(binaryExpressionAccumulator.getOperatorSymbol());
    if (operatorChangedSinceLastAccumulation || isFirstBinaryExprInSubtree) {
      // if the operator changed and we could simplify the accumulated values, we need to replace the subtree starting
      // at node lastVisitedBinaryExp with the simplified subtree
      if (operatorChangedSinceLastAccumulation && binaryExpressionAccumulator.subtreeIsSimplified()) {
        arithmeticExpr.replaceChild(binaryExpressionAccumulator.lastVisitedBinaryExp,
                                    binaryExpressionAccumulator.getSimplifiedSubtree());
      }
      // if operator is non-commutative -> return from function as operator is unsupported
      if (!BinaryExpressionAcc::isSupportedOperator(currentOperator->getOperatorSymbol())) {
        return;
      }
      // if this is a supported operator, remember the operator's symbol
      binaryExpressionAccumulator.removeOperandsAndSetNewSymbol(currentOperator->getOperatorSymbol());
      // remember this first binary expression as we may need to re-attach its children later
      binaryExpressionAccumulator.setFirstVisitedSubtree(&arithmeticExpr);
    }

    // if this is part of a nested binary expression, one of both operands will be a BinaryExpr itself: as its operands
    // were already added when visiting this BinaryExpr, do not add it again
    binaryExpressionAccumulator.addOperands({lhsOperand, rhsOperand});
    binaryExpressionAccumulator.setLastVisitedSubtree(&arithmeticExpr);

    // if we have collected something in binaryExpressionAccumulator and none of the parents is another BinaryExpr,
    // we need to decide whether it makes sense so replace the subtree (if expression could be simplified) or not
    auto noneOfTheParentsIsABinaryExpr = [](AbstractExpr *expr) {
      for (auto &p : expr->getParentsNonNull()) {
        if (dynamic_cast<AbstractBinaryExpr *>(p)) return false;
      }
      return true;
    };
    // We can consider the subtree as simplified if the number of leaf nodes (AbstractLiterals or Variables) decreased.
    // Only replace current subtree by simplified one if all of:
    //   i. there are any accumulated values (should always be the case if binary expressions were handled properly)
    //   ii. there are no more binary expressions following, i.e., in a higher level of the tree
    //   iii. the accumulation was useful, i.e., reduced the total number of leaf nodes
    if (binaryExpressionAccumulator.containsOperands()
        && noneOfTheParentsIsABinaryExpr(&arithmeticExpr)
        && binaryExpressionAccumulator.subtreeIsSimplified()) {
      auto treeRoot = binaryExpressionAccumulator.getSimplifiedSubtree();
      // replace the current arithmetic expression by the one generated out of the accumulated (and simplified) operands
      arithmeticExpr.getOnlyParent()->replaceChild(arithmeticExpr.castTo<AbstractNode>(), treeRoot);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(ArithmeticExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);
  handleBinaryExpression(elem);
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getLeft());
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(LogicalExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);

  // Apply Boolean simplification rules, for example:
  //   a.) <anything> AND true  ==  <anything>
  //   b)  <anything> OR true  ==  true
  // We only need to consider the case where exactly one of the two operands is a LiteralBool because the case of two
  // known Boolean values is already handled by handleBinaryExpression. Also, if none of the operands is a literal, we
  // cannot perform any simplifications.
  auto lhs = dynamic_cast<LiteralBool *>(elem.getLeft());
  auto lhsIsBool = lhs!=nullptr;
  auto rhs = dynamic_cast<LiteralBool *>(elem.getRight());
  auto rhsIsBool = rhs!=nullptr;

  if ((lhsIsBool && !rhsIsBool) || (rhsIsBool && !lhsIsBool)) {  // if one of the operands is a Boolean
    auto determineBooleanNonBooleanOperand =
        [](LiteralBool *lhs, AbstractExpr *lhsAlt, LiteralBool *rhs, AbstractExpr *rhsAlt)
            -> std::pair<LiteralBool *, AbstractExpr *> {
          return (lhs!=nullptr) ? std::make_pair(lhs, rhsAlt) : std::make_pair(rhs, lhsAlt);
        };
    auto[booleanOperand, nonBooleanOperand] = determineBooleanNonBooleanOperand(
        lhs, elem.getLeft(), rhs, elem.getRight());

    // <anything> AND true  ==  <anything>
    // <anything> OR false  ==  <anything>
    // <anything> XOR false  ==  <anything>
    if ((elem.getOp()->equals(logicalAnd) && booleanOperand->getValue())
        || (elem.getOp()->equals(logicalOr) && !booleanOperand->getValue())
        || (elem.getOp()->equals(logicalXor) && !booleanOperand->getValue())) {
      nonBooleanOperand->removeFromParents();
      elem.getOnlyParent()->replaceChild(&elem, nonBooleanOperand);
      nodesQueuedForDeletion.push_back(&elem);
    } else if (elem.getOp()->equals(logicalAnd) && !booleanOperand->getValue()) {  // <anything> AND false  ==  false
      elem.getOnlyParent()->replaceChild(&elem, new LiteralBool(false));
      nodesQueuedForDeletion.push_back(nonBooleanOperand);
    } else if (elem.getOp()->equals(logicalOr) && booleanOperand->getValue()) {   // <anything> OR true  ==  true
      elem.getOnlyParent()->replaceChild(&elem, new LiteralBool(true));
      nodesQueuedForDeletion.push_back(nonBooleanOperand);
    } else if (elem.getOp()->equals(logicalXor)
        && booleanOperand->getValue()) {  // <anything> XOR true  ==  NOT <anything>
      nonBooleanOperand->removeFromParents();
      auto uexp = new UnaryExpr(negation, nonBooleanOperand);
      elem.getOnlyParent()->replaceChild(&elem, uexp);
      nodesQueuedForDeletion.push_back(&elem);
    }
  } else {
    // handles the case where both operands are known or neither one and tries to simplify nested logical expressions
    handleBinaryExpression(elem);
  }
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getLeft());
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
  // check if the unary expression can be evaluated
  if (dynamic_cast<AbstractLiteral *>(elem.getRight())) {
    // if operand value is known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&elem, getTransformedVariableMap());
    auto parent = elem.getOnlyParent();
    parent->replaceChild(&elem, result.front());
    nodesQueuedForDeletion.push_back(&elem);
  }
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(Block &elem) {
  Visitor::visit(elem);
  // If all Block's statements are marked for deletion, mark it as evaluated to notify its parent.
  // The parent is then responsible to decide whether it makes sense to delete this Block or not.

  // check if there is any statement within this Block that is not marked for deletion
  bool allStatementsInBlockAreMarkedForDeletion = true;
  for (auto &statement : elem.getStatements()) {
    if (removableNodes.count(statement)==0) {
      allStatementsInBlockAreMarkedForDeletion = false;
    }
    // clean up temporary results from children in removableNodes map
    removableNodes.erase(statement);
  }
  // if all statements of this Block are marked for deletion, we can mark the Block as removable
  // -> let the Block's parent decide whether to keep this empty Block or not
  if (allStatementsInBlockAreMarkedForDeletion) markNodeAsRemovable(&elem);

  cleanUpAfterStatementVisited(&elem, false);
}

void CompileTimeExpressionSimplifier::visit(Call &elem) {
  Visitor::visit(elem);
  // TODO(pjattke): implement me!
  //  We can replace the call node by the result of the inner function's return value.
  //  For example:
  //
  //    int computeX(int input) {
  //      ...
  //      int z = computeZ(seed);   -->  int computeZ(int seed) { int v = genRandomNum(seed); return v + 32; }
  //      ...
  //    }
  //
  //  is converted into:
  //
  //    int computeX(int input) {
  //      ...
  //      int z = genRandomNum(seed) + 32;
  //      ...
  //    }
}

void CompileTimeExpressionSimplifier::visit(ParameterList &elem) {
  Visitor::visit(elem);
  // if all of the FunctionParameter children are marked as removable, mark this node as removable too
  bool allFunctionParametersAreRemovable = true;
  for (auto &fp : elem.getParameters()) {
    if (!valueIsKnown(fp)) allFunctionParametersAreRemovable = false;
    // clean up temporary results from children in removableNodes map
    removableNodes.erase(fp);
  }
  if (allFunctionParametersAreRemovable) {
    markNodeAsRemovable(&elem);
  }
  cleanUpAfterStatementVisited(&elem, false);
}

void CompileTimeExpressionSimplifier::visit(Function &elem) {
  Visitor::visit(elem);
  // This node is marked as removable if both the ParameterList (implies all of the FunctionParameters) and the Block
  // (implies all of the statements) are removable. This mark is only relevant in case that this Function is included
  // in a Call statement because Call statements can be replaced by inlining the Function's computation.
  if (valueIsKnown(elem.getParameterList()) && valueIsKnown(elem.getBody())) {
    markNodeAsRemovable(&elem);
  }
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getParameterList());
  removableNodes.erase(elem.getBody());
  cleanUpAfterStatementVisited(&elem, false);
}

void CompileTimeExpressionSimplifier::visit(FunctionParameter &elem) {
  Visitor::visit(elem);

  // This simplifier does not care about the variable's datatype, hence we can mark this node as removable. This mark is
  // only relevant in case that this FunctionParameter is part of a Function that is included into a Call
  // statement because Call statements can be replaced by inlining the Function's computation.
  markNodeAsRemovable(&elem);
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
      // recursively remove the Else-branch (sanity-check, may not necessarily exist)
      if (elem.getElseBranch()!=nullptr) {
        nodesQueuedForDeletion.push_back(elem.getElseBranch());
        // we also unlink it from the If statement such that it will not be visited
        elem.removeChild(elem.getElseBranch(), true);
      }
    } else {  // the Else-branch is always executed
      // recursively remove the Then-branch (always exists)
      nodesQueuedForDeletion.push_back(elem.getThenBranch());
      // negate the condition and delete the conditions stored value (is now invalid)
      auto condition = elem.getCondition();
      auto newCondition = new UnaryExpr(UnaryOp::negation, condition);
      removableNodes.erase(condition);
      // replace the If statement's Then branch by the Else branch
      elem.removeChild(elem.getThenBranch(), true);
      elem.setAttributes(newCondition, elem.getElseBranch(), nullptr);
    }

    // continue visiting the remaining branches: the condition will be visited again, but that's ok
    Visitor::visit(elem);

    // we can remove the whole If statement if...
    if ( // the Then-branch is always executed and it is empty after simplification (thus queued for deletion)
        (thenAlwaysExecuted && isQueuedForDeletion(elem.getThenBranch()))
            // the Then-branch is never executed but there is no Else-branch
            || (!thenAlwaysExecuted && elem.getElseBranch()==nullptr)
                // the Else-branch is always executed but it is empty after simplification  (thus queued for deletion)
            || (!thenAlwaysExecuted && isQueuedForDeletion(elem.getElseBranch()))) {
      // enqueue the If statement and its children for deletion
      nodesQueuedForDeletion.push_back(&elem);
    }
  }
    // ================
    // Case 2: Condition's evaluation result is UNKNOWN at compile-time
    // -> rewrite variables values that are modified in either one or both of the If statement's branches such that the
    //    variable's value depends on the If statement's condition evaluation result.
    // ================
  else { // if we don't know the evaluation result of the If statement's condition -> rewrite the If statement
    // create a copy of the variableValues map and removableNodes map
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

      // Determine if an If statement-dependent value needs to be assigned to the variable.
      // The following approach only rewrites those variables that were modified in the Then-branch, the Else-branch, or
      // both branches. It does, however, drop new variables (i.e., variable declarations) that happened in one of the
      // branches. Those variables are anyway out of scope when leaving the branch, so there's no need to store their
      // value in variableValues map.
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
    nodesQueuedForDeletion.push_back(&elem);
  }
  // clean up temporary results from children in removableNodes map
  removableNodes.erase(elem.getCondition());
  removableNodes.erase(elem.getThenBranch());
  removableNodes.erase(elem.getElseBranch());

  cleanUpAfterStatementVisited(&elem, false);
}

void CompileTimeExpressionSimplifier::visit(While &elem) {
  // visit the condition only
  elem.getCondition()->accept(*this);

  // check if we know the While condition's truth value at compile time
  auto conditionValue = dynamic_cast<LiteralBool *>(elem.getCondition());
  if (conditionValue!=nullptr && !conditionValue->getValue()) {
    // While is never executed: remove While-loop including contained statements
    cleanUpAfterStatementVisited(reinterpret_cast<AbstractNode *>(&elem), true);
    return;
  }

  // visit body (and condition again, but that's acceptable)
  Visitor::visit(elem);

  cleanUpAfterStatementVisited(reinterpret_cast<AbstractNode *>(&elem), false);
}

void CompileTimeExpressionSimplifier::visit(For &elem) {
  Visitor::visit(elem);
  cleanUpAfterStatementVisited(reinterpret_cast<AbstractNode *>(&elem), false);
}

void CompileTimeExpressionSimplifier::visit(Return &elem) {
  Visitor::visit(elem);
  // simplify return expression: replace each evaluated expression by its evaluation result
  bool allValuesAreKnown = true;
  for (auto &returnExpr : elem.getReturnExpressions()) {
    if (!valueIsKnown(returnExpr)) allValuesAreKnown = false;
    // clean up temporary results from children in removableNodes map
    removableNodes.erase(returnExpr);
  }
  // marking this node as removable is only useful for the case that this Return belongs to a nested Function that can
  // be replaced by inlining the return expressions into the position where the function is called
  if (allValuesAreKnown) markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, false);
}

// =====================
// Helper methods
// =====================

bool CompileTimeExpressionSimplifier::valueIsKnown(AbstractNode *node) {
  // A value is considered as known if...
  // i.) it is a Literal
  if (dynamic_cast<AbstractLiteral *>(node)!=nullptr) return true;

  // ii.) it is a variable with a known value (in variableValues)
  auto variableValueIsKnown = false;
  if (auto abstractExprAsVariable = dynamic_cast<Variable *>(node)) {
    // check that the variable has a value
    return variableValues.count(abstractExprAsVariable->getIdentifier()) > 0
        // and its value is not symbolic (i.e., contains no variables for which the value is unknown)
        && variableValues.at(abstractExprAsVariable->getIdentifier())->getVariableIdentifiers().empty();
  }
  // ii.) or the node is removable (i.e., its value is in removableNodes)
  return variableValueIsKnown || removableNodes.count(node) > 0;
}

void CompileTimeExpressionSimplifier::markNodeAsRemovable(AbstractNode *node) {
  if (removableNodes.count(node)==0) removableNodes.insert(node);
}

std::vector<AbstractLiteral *> CompileTimeExpressionSimplifier::evaluateNodeRecursive(
    AbstractNode *node, std::unordered_map<std::string, AbstractLiteral *> valuesOfVariables) {
  // clean up the EvaluationVisitor from any previous run
  evalVisitor.reset();

  // perform evaluation by passing the required parameter values
  evalVisitor.updateVarValues(std::move(valuesOfVariables));
  node->accept(evalVisitor);

  // retrieve and return results
  return evalVisitor.getResults();
}

AbstractExpr *CompileTimeExpressionSimplifier::getFirstValue(AbstractNode *node) {
  // if node is a Literal -> return node itself
  if (auto nodeAsLiteral = dynamic_cast<AbstractLiteral *>(node)) {
    return nodeAsLiteral;
  }
  // if node is a variable -> search the variable's value in the map of known variable values
  auto nodeAsVariable = dynamic_cast<Variable *>(node);
  if (nodeAsVariable!=nullptr && variableValues.count(nodeAsVariable->getIdentifier()) > 0) {
    return variableValues.at(nodeAsVariable->getIdentifier());
  }
  // in any other case: throw an error
  throw std::invalid_argument("Cannot determine value for node " + node->getUniqueNodeId() + ".");
}

std::unordered_map<std::string, AbstractLiteral *> CompileTimeExpressionSimplifier::getTransformedVariableMap() {
  std::unordered_map<std::string, AbstractLiteral *> variableMap;
  for (auto &[k, v] : variableValues) {
    if (auto varAsLiteral = dynamic_cast<AbstractLiteral *>(v)) {
      variableMap[k] = varAsLiteral;
    }
  }
  return variableMap;
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
                                            ArithmeticOp::subtraction,
                                            condition->clone(false)->castTo<AbstractExpr>());
    // case: trueValue == 0 && falseValue != 0 => value is 0 if the condition is True
    // -> return (1-b)*falseValue
    return new ArithmeticExpr(factorIsFalse, ArithmeticOp::multiplication, falseValue);
  } else if (falseValueIsNull) {
    // factorIsTrue = ifStatementCondition
    auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
    // case: trueValue != 0 && falseValue == 0 => value is 0 if the condition is False
    // -> return condition * trueValue
    return new ArithmeticExpr(factorIsTrue, ArithmeticOp::multiplication, trueValue);
  }

  // default case: trueValue != 0 && falseValue != 0 => value is changed in both branches of If statement
  // -> return condition*trueValue + (1-b)*falseValue.
  // factorIsTrue = ifStatementCondition
  auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
  // factorIsFalse = [1-ifStatementCondition]
  auto factorIsFalse = new ArithmeticExpr(new LiteralInt(1),
                                          ArithmeticOp::subtraction,
                                          condition->clone(false)->castTo<AbstractExpr>());
  return new ArithmeticExpr(
      new ArithmeticExpr(factorIsTrue,
                         ArithmeticOp::multiplication,
                         trueValue->clone(false)->castTo<AbstractExpr>()),
      ArithmeticOp::addition,
      new ArithmeticExpr(factorIsFalse,
                         ArithmeticOp::multiplication,
                         falseValue->clone(false)->castTo<AbstractExpr>()));
}

void CompileTimeExpressionSimplifier::cleanUpAfterStatementVisited(AbstractNode *statement,
                                                                   bool enqueueStatementForDeletion) {
  if (enqueueStatementForDeletion) {
    // mark this statement for deletion as we don't need it anymore
    nodesQueuedForDeletion.push_back(statement);
  }
  // free the accumulator of binary expressions
  binaryExpressionAccumulator.reset();
}

void CompileTimeExpressionSimplifier::addVariableValue(const std::string &variableIdentifier,
                                                       AbstractExpr *valueAnyLiteralOrAbstractExpr) {
  valueAnyLiteralOrAbstractExpr->removeFromParents();
  variableValues[variableIdentifier] = valueAnyLiteralOrAbstractExpr->clone(false)
      ->castTo<AbstractExpr>();
}

bool CompileTimeExpressionSimplifier::isQueuedForDeletion(const AbstractNode *node) {
  return std::find(nodesQueuedForDeletion.begin(), nodesQueuedForDeletion.end(), node)
      !=nodesQueuedForDeletion.end();
}

