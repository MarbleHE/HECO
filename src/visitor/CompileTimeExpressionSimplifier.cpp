#include <unordered_set>
#include <DotPrinter.h>
#include "CompileTimeExpressionSimplifier.h"
#include "BinaryExpr.h"
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

void CompileTimeExpressionSimplifier::handleBinaryExpressions(AbstractNode &binaryExpr,
                                                              AbstractExpr *leftOperand,
                                                              AbstractExpr *rightOperand) {
  auto leftValueIsKnown = valueIsKnown(leftOperand);
  auto rightValueIsKnown = valueIsKnown(rightOperand);
  if (leftValueIsKnown && rightValueIsKnown) {
    // if both operand values are known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&binaryExpr, getTransformedVariableMap());
    storeEvaluatedNode(&binaryExpr, std::vector<AbstractExpr *>(result.begin(), result.end()));
  } else if (leftValueIsKnown || rightValueIsKnown) {
    // if only one of both is known -> simplify expression by replacing operand's value by the evaluation result
    auto concernedOperand = leftValueIsKnown ? leftOperand : rightOperand;
    auto newValue = getFirstValue(concernedOperand);
    binaryExpr.removeChildBilateral(concernedOperand);
    binaryExpr.addChild(newValue);
  }
  // clean up temporary results from children in evaluatedNodes map
  evaluatedNodes.erase(leftOperand);
  evaluatedNodes.erase(rightOperand);
  evaluatedNodes.erase(binaryExpr.getChildAtIndex(1)); // operator
}

void CompileTimeExpressionSimplifier::visit(BinaryExpr &elem) {
  Visitor::visit(elem);
  handleBinaryExpressions(elem, elem.getLeft(), elem.getRight());
}

void CompileTimeExpressionSimplifier::visit(LogicalExpr &elem) {
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

void CompileTimeExpressionSimplifier::visit(Variable &elem) {
  Visitor::visit(elem);
  // if the variable's value is known yet mark the node as evaluated
  if (valueIsKnown(&elem)) {
    storeEvaluatedNode(&elem, getFirstValue(&elem));
  } else if (variableValues.count(elem.getIdentifier()) > 0) {
    auto variableParent = elem.getParentsNonNull().front();
    elem.isolateNode();
    nodesQueuedForDeletion.push(&elem);
    variableParent->addChild(
        variableValues.at(elem.getIdentifier())->clone(false)->castTo<AbstractExpr>(),
        true);
  }
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
    std::unordered_map<AbstractNode *, std::vector<AbstractExpr *>> originalEvaluatedNodes(evaluatedNodes);

    // visit the thenBranch and store its modifications
    elem.getThenBranch()->accept(*this);
    std::unordered_map<std::string, AbstractExpr *> variableValuesAfterVisitingThen(variableValues);
    std::unordered_map<AbstractNode *, std::vector<AbstractExpr *>> evalutedNodesAfterVisitingThen(evaluatedNodes);

    // check if there is an Else-branch that we need to visit
    if (elem.getElseBranch()!=nullptr) {
      // restore the original maps via copy assignments prior visiting Else-branch
      variableValues = originalVariableValues;
      evaluatedNodes = originalEvaluatedNodes;

      // visit the Else-branch
      elem.getElseBranch()->accept(*this);
    }

    // restore the original maps via copy assignments
//    variableValues = originalVariableValues;
//    evaluatedNodes = originalEvaluatedNodes;

    // up to this point (and beyond), the Else-branch's modifications are in variableValues
    // rewrite those entries that were modified in either one or both maps
    IfStatementResolver isr(elem.getCondition());
    for (auto &[variableIdentifier, originalValue] : originalVariableValues) {

      auto thenBranchValue = variableValuesAfterVisitingThen.at(variableIdentifier);
      auto thenBranchModifiedValue = (thenBranchValue!=originalValue);

      bool elseBranchModifiedValue = false;
      AbstractExpr *elseBranchValue = nullptr;
      if (elem.getElseBranch()!=nullptr) {
        elseBranchValue = variableValues.at(variableIdentifier);
        elseBranchModifiedValue = (elseBranchValue!=originalValue);
      }

      if (thenBranchModifiedValue && elseBranchModifiedValue) {
        originalVariableValues[variableIdentifier] = isr.generateIfDependentValue(thenBranchValue, elseBranchValue);
      } else if (thenBranchModifiedValue) {
        originalVariableValues[variableIdentifier] = isr.generateIfDependentValue(thenBranchValue, originalValue);
      } else if (elseBranchModifiedValue) {
        originalVariableValues[variableIdentifier] = isr.generateIfDependentValue(originalValue, elseBranchValue);
      } // otherwise neither one of the two branches modified the variable's value and we can keep it like it is

      // restore the original maps via copy assignments
      variableValues = originalVariableValues;
      evaluatedNodes = originalEvaluatedNodes;
    }


//    // set resolveIfStatementActive = true; b = [condition], notB = (1-[condition])
//    ifResolverData.push(new IfStatementResolver(elem.getCondition()));
//
//    // visit the thenBranch
//    elem.getThenBranch()->accept(*this);
//    // if exists, visit the elseBranch
//    if (elem.getElseBranch()!=nullptr) {
//      ifResolverData.top()->setActiveBranch(false);
//      elem.getElseBranch()->accept(*this);
//    }
//    // TODO(pjattke): modify visits of AST objects to consider resolveIfStatementActive
//
//    //  Generate If-dependent assignment values by using map ifStatementVariableValues
//    //  if either one of the pair's values (first, second) is empty, we need to pass the value's current value instead
//    //  such that the value stays the same
//    //  Consider following example where 22 is the variable's original value that needs to be preserved if not [a>42]:
//    //    int i = 22; if (a > 42) { i = 111; }  --> int i = [a>42]*111 + [1-[a>42]]*22;
//    auto currentIfResolverStruct = ifResolverData.top();
//    for (auto &[varIdentifier, valueThenElse] : currentIfResolverStruct->getIfStatementVariableValues()) {
//      //
//      if (valueThenElse.first==nullptr || valueThenElse.second==nullptr) {
//        // retrieve old value
//        AbstractExpr *oldValue =
//            (variableValues.count(varIdentifier) > 0) ? variableValues.at(varIdentifier) : new Variable(varIdentifier);
//        // save new value that depends on the If statement's condition
//        if (valueThenElse.first==nullptr) {
//          variableValues[varIdentifier] =
//              ifResolverData.top()->generateIfDependentValue(oldValue, valueThenElse.second);
//        } else if (valueThenElse.second==nullptr) {
//          variableValues[varIdentifier] = ifResolverData.top()->generateIfDependentValue(valueThenElse.first, oldValue);
//        }
//      } else {
//        variableValues[varIdentifier] =
//            ifResolverData.top()->generateIfDependentValue(valueThenElse.first, valueThenElse.second);
//      }
//    }
//    // remove the ifStatementResolverData object
//    ifResolverData.pop();

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

  // if there was only one statement and this statement was evaluable, mark this Return node as evaluable
  // (If this Return is part of a Function that is nested into a Call and this Return is evaluable, then we can replace
  // the Call by the evaluated value. But only if there is a single return value because our AST cannot represent the
  // assignment of multiple return values yet.)
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
        && variableValues.at(abstractExprAsVariable->getIdentifier())->getVariableIdentifiers().size()==0;
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
