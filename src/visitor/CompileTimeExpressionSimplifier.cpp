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
#include "CallExternal.h"

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
    nodesQueuedForDeletion.pop_front();
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
  cleanUpAfterStatementVisited(elem.AbstractStatement::castTo<AbstractStatement>(), false);
}

// =====================
// AST objects that are trivially evaluable (AST leaf nodes)
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

void CompileTimeExpressionSimplifier::visit(Variable &elem) {
  Visitor::visit(elem);
  if (variableValues.count(elem.getIdentifier()) > 0) {
    // if we know the variable's value (i.e., its value is either any subtype of AbstractLiteral or a AbstractExpr if
    // this is a symbolic value that defines on other variables), we can replace this variable node
    auto variableParent = elem.getParentsNonNull().front();
    auto newValue = variableValues.at(elem.getIdentifier())->clone(false)->castTo<AbstractExpr>();
    variableParent->replaceChild(&elem, newValue);

    // if this newValue is an AbstractBinaryExpr by itself, enqueue its operands
    if (auto bexp = dynamic_cast<AbstractBinaryExpr *>(newValue)) {
      binaryExpressionAccumulator.addOperands({bexp->getLeft(), bexp->getRight()});
      binaryExpressionAccumulator.setLastVisitedSubtree(bexp);
    }
  }
}

/** @} */ // End of visit leaves group


// =====================
// Simplifiable Statements
// =====================

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  Visitor::visit(elem);

//  // stop execution here if there is an initializer but the initializer's value is not known
//  if (elem.getInitializer()!=nullptr && !valueIsKnown(elem.getInitializer())) {
//    // clean up temporary results from children in evaluatedNodes map
//    evaluatedNodes.erase(elem.getInitializer());
//    return;
//  }
//
//  AbstractExpr *variableValue;
//  if (elem.getInitializer()==nullptr) {
//    // If this variable was declared but not initialized, then we store the variable's default value (e.g., 0 for int,
//    // 0.0f for float) in the map. This is required because if this variable will be changed in an If statement, we need
//    // to have the variable's original value for rewriting the If statement.
//    variableValue = getDefaultVariableInitializationValue(elem.getDatatype()->getType());
//  } else if (valueIsKnown(elem.getInitializer())) {
//    // If the initializer's value is known, we assign its value to the variable this VarDecl refers to.
//    variableValue = getFirstValue(elem.getInitializer());
//  } else {
//    variableValue = elem.getInitializer();
//  }

  AbstractExpr *variableValue;
  if (elem.getInitializer()==nullptr) {
    variableValue = getDefaultVariableInitializationValue(elem.getDatatype()->getType());
  } else {
    variableValue = elem.getInitializer();
  }

  // store the variable's value
  addVariableValue(elem.getVarTargetIdentifier(), variableValue);

  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getInitializer());

  // mark this statement as evaluated to notify its parent that this can be considered as evaluated
  storeEvaluatedNode(&elem, variableValue);

  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::visit(VarAssignm &elem) {
  Visitor::visit(elem);

  addVariableValue(elem.getVarTargetIdentifier(), elem.getValue());

  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getValue());

  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::handleBinaryExpressions(AbstractBinaryExpr &arithmeticExpr,
                                                              AbstractExpr *leftOperand,
                                                              AbstractExpr *rightOperand) {
  auto leftValueIsKnown = valueIsKnown(leftOperand);
  auto rightValueIsKnown = valueIsKnown(rightOperand);
  if (leftValueIsKnown && rightValueIsKnown) {
    // if both operand values are known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&arithmeticExpr, getTransformedVariableMap());
    if (arithmeticExpr.getParentsNonNull().size() > 1)
      throw std::logic_error("Unexpected: Node (" + arithmeticExpr.getUniqueNodeId() + ") has more than one parent.");
    if (result.size() > 1)
      throw std::logic_error("Unexpected: Evaluation result contains more than one value.");
    arithmeticExpr.getParentsNonNull().front()->replaceChild(&arithmeticExpr, result.front());
    nodesQueuedForDeletion.push_back(&arithmeticExpr);
  } else {
    // update accumulator
    // - check if the operator changed
    auto currentOperator = arithmeticExpr.getOp();
    if (!binaryExpressionAccumulator.containsOperands()) {  // this is the first binary expression in this subtree
      binaryExpressionAccumulator.setOperator(currentOperator->getOperatorSymbol());
    } else if (!currentOperator->equals(binaryExpressionAccumulator.getOperatorSymbol())) {  // operator changed
      // check if we could simplify anything, if yes: replace current subtree by simplified one
      if (binaryExpressionAccumulator.subtreeIsSimplified()) {
        arithmeticExpr.replaceChild(binaryExpressionAccumulator.lastVisitedSubtree,
                                    binaryExpressionAccumulator.getSimplifiedSubtree());
      }
      // use accumulation only for commutative operations -> return from function if operator is unsupported
      if (!binaryExpressionAccumulator.isSupportedOperator(currentOperator->getOperatorSymbol())) return;
      binaryExpressionAccumulator.removeOperandsAndSetNewSymbol(currentOperator->getOperatorSymbol());
    }
    // - if this is part of a nested binary expression, one of both operands will be a BinaryExpr itself - as its operands
    //   were already added when visiting this BinaryExpr, do not add it again
    binaryExpressionAccumulator.addOperands({leftOperand, rightOperand});
    binaryExpressionAccumulator.setLastVisitedSubtree(&arithmeticExpr);

    // if we have collected something in binaryExpressionAccumulator and none of the parents is another BinaryExpr,
    // we need to reconstruct the subtree by using the accumulated values
    auto noneOfTheParentsIsABinaryExpr = [](AbstractExpr &expr) {
      for (auto &p : expr.getParentsNonNull()) { if (dynamic_cast<AbstractBinaryExpr *>(p)) return false; }
      return true;
    };
    // we can consider the subtree as simplified if the number of leaf nodes (AbstractLiterals or Variables) decreased
    // only replace current subtree by simplified one if all of:
    // i. there are any accumulated values at all (should always be the case)
    // ii. there are no more binary expressions following in a higher level of the tree
    // iii. the accumulation was useful, i.e., reduced the number of leaf nodes
    if (binaryExpressionAccumulator.containsOperands() && noneOfTheParentsIsABinaryExpr(arithmeticExpr)
        && binaryExpressionAccumulator.subtreeIsSimplified()) {
      auto treeRoot = binaryExpressionAccumulator.getSimplifiedSubtree();
      arithmeticExpr.getParentsNonNull().front()->replaceChild(arithmeticExpr.castTo<AbstractNode>(), treeRoot);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(ArithmeticExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);
  handleBinaryExpressions(elem, elem.getLeft(), elem.getRight());
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getLeft());
  evaluatedNodes.erase(elem.getRight());
  evaluatedNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(LogicalExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);
  handleBinaryExpressions(elem, elem.getLeft(), elem.getRight());
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getLeft());
  evaluatedNodes.erase(elem.getRight());
  evaluatedNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
  // check if the unary expression can be evaluated
  if (dynamic_cast<AbstractLiteral *>(elem.getRight())) {
    // if operand value is known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&elem, getTransformedVariableMap());
    auto parent = elem.getParentsNonNull().front();
    parent->replaceChild(&elem, result.front());
    nodesQueuedForDeletion.push_back(&elem);
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

  cleanUpAfterStatementVisited(&elem, false);
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
  cleanUpAfterStatementVisited(&elem, false);
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

  cleanUpAfterStatementVisited(&elem, false);
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
        nodesQueuedForDeletion.push_back(elem.getElseBranch());
        // we also unlink it from the If statement such that it will not be visited
        elem.removeChildBilateral(elem.getElseBranch());
      }
    } else {  // the Else-branch is always executed
      // recursively remove the Then-branch (always exists)
      nodesQueuedForDeletion.push_back(elem.getThenBranch());
      // negate the condition and delete the conditions stored value (is now invalid)
      auto condition = elem.getCondition();
      auto newCondition = new UnaryExpr(UnaryOp::negation, condition);
      evaluatedNodes.erase(condition);
      // replace the If statement's Then branch by the Else branch
      elem.removeChildBilateral(elem.getThenBranch());
      elem.setAttributes(newCondition, elem.getElseBranch(), nullptr);
    }

    // continue visiting the remaining branches
    // (the condition will be visited again, but that's acceptable)
    Visitor::visit(elem);

    // check if we can remove the whole If statement
    bool wholeIfStmtCanBeDeleted = false;
    if (thenAlwaysExecuted && isQueuedForDeletion(elem.getThenBranch())) {
      wholeIfStmtCanBeDeleted = true;
    } else if (elem.getElseBranch()==nullptr || isQueuedForDeletion(elem.getElseBranch())) {
      // The whole If statement can be deleted if there exists no Else branch (If body is not executed at all) or if the
      // Else-branch is queued for deletion
      wholeIfStmtCanBeDeleted = true;
    }
    // enqueue the If statement and its children for deletion
    if (wholeIfStmtCanBeDeleted) nodesQueuedForDeletion.push_back(&elem);
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
    nodesQueuedForDeletion.push_back(&elem);
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(elem.getCondition());
  evaluatedNodes.erase(elem.getThenBranch());
  evaluatedNodes.erase(elem.getElseBranch());

  cleanUpAfterStatementVisited(&elem, false);
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
  cleanUpAfterStatementVisited(reinterpret_cast<AbstractNode *>(&elem), false);
}

void CompileTimeExpressionSimplifier::visit(Return &elem) {
  Visitor::visit(elem);
  // simplify return expression: replace each evaluated expression by its evaluation result
  bool allValuesAreKnown = true;
  for (auto &returnExpr : elem.getReturnExpressions()) {
    if (!valueIsKnown(returnExpr)) allValuesAreKnown = false;
    // clean up temporary results from children in evaluatedNodes map
    evaluatedNodes.erase(returnExpr);
  }
  if (allValuesAreKnown) storeEvaluatedNode(&elem, elem.getReturnExpressions());
  cleanUpAfterStatementVisited(&elem, false);
}

// =====================
// Helper methods
// =====================

bool CompileTimeExpressionSimplifier::valueIsKnown(AbstractNode *abstractExpr) {
  // A value is considered as known if...
  // i.) it is a Literal
  if (dynamic_cast<AbstractLiteral *>(abstractExpr)!=nullptr) return true;

  // ii.) it is a variable with a known value (in variableValues)
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
  // if node is a Literal -> return node itself
  if (auto nodeAsLiteral = dynamic_cast<AbstractLiteral *>(node)) {
    return nodeAsLiteral;
  }

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
                                            ArithmeticOp::subtraction,
                                            condition->clone(false)->castTo<AbstractExpr>());
    // case: trueValue == 0 && falseValue != 0 => value is 0 if the condition is True -> return (1-b)*falseValue
    return new ArithmeticExpr(factorIsFalse, ArithmeticOp::multiplication, falseValue);
  } else if (falseValueIsNull) {
    // factorIsTrue = ifStatementCondition
    auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
    // case: trueValue != 0 && falseValue == 0 => value is 0 if the condition is False -> return condition * trueValue
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

void CompileTimeExpressionSimplifier::cleanUpAfterStatementVisited(AbstractNode *node,
                                                                   bool enqueueStatementForDeletion) {
  if (enqueueStatementForDeletion) {
    // mark this statement for deletion as we don't need it anymore
    nodesQueuedForDeletion.push_back(node);
  }

  // free the accumulator of binary expressions
  binaryExpressionAccumulator.clear();
}

void CompileTimeExpressionSimplifier::addVariableValue(const std::string &variableIdentifier,
                                                       AbstractExpr *valueAnyLiteralOrAbstractExpr) {
  valueAnyLiteralOrAbstractExpr->removeFromParents();
  variableValues[variableIdentifier] = valueAnyLiteralOrAbstractExpr;
}

bool CompileTimeExpressionSimplifier::isQueuedForDeletion(const AbstractNode *node) {
  return std::find(nodesQueuedForDeletion.begin(), nodesQueuedForDeletion.end(), node)
      !=nodesQueuedForDeletion.end();
}
