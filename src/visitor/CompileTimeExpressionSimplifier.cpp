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
 *    These methods add a clone of the node (with its value) as evaluation result because a statement like
 *    "bool b = true;" will be simplified by copying the variable-value pair (b, *LiteralBool(true)) into
 *    variableValues. After finishing the simplification traversal, the node VarDecl of "bool b = true;" and all of its
 *    children, including LiteralBool(true), will be deleted. Hence, looking up any value in variableValues as needed by
 *    tests is not possible after ending the AST simplification traversal. Because of that, a clone is added here.
 *  @{
 */

void CompileTimeExpressionSimplifier::visit(LiteralBool &elem) {
  storeEvaluatedNode(&elem, elem.clone(false));
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralInt &elem) {
  storeEvaluatedNode(&elem, elem.clone(false));
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralString &elem) {
  storeEvaluatedNode(&elem, elem.clone(false));
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralFloat &elem) {
  storeEvaluatedNode(&elem, elem.clone(false));
  Visitor::visit(elem);
}

/** @} */ // End of visit leaves group


// =====================
// Simplifiable Statements
// =====================

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  Visitor::visit(elem);
  if (elem.getInitializer()==nullptr) { // if this variable was declared but not initialized
    // then we store the variable's default value in the map
    variableValues
        .emplace(elem.getVarTargetIdentifier(), getDefaultVariableInitializationValue(elem.getDatatype()->getType()));
    // mark this statement for deletion as we don't need it anymore (we don't care about the lost type information)
    nodesQueuedForDeletion.push(&elem);
    // mark this statement as evaluated to notify its parent that this can be considered as evaluated
    storeEvaluatedNode(&elem, nullptr);
  } else if (valueIsKnown(elem.getInitializer())) { // if the initializer's value is known
    // store the variable's value in the map
    variableValues.emplace(elem.getVarTargetIdentifier(), getFirstValue(elem.getInitializer()));
    // mark this statement for deletion as we don't need it anymore
    nodesQueuedForDeletion.push(&elem);
    // mark this statement as evaluated to notify its parent about a known value
    storeEvaluatedNode(&elem, getFirstValue(elem.getInitializer()));
  }
}

void CompileTimeExpressionSimplifier::visit(VarAssignm &elem) {
  Visitor::visit(elem);
  if (!ifResolverData.empty()) {
    if (valueIsKnown(elem.getValue())) {
      ifResolverData.top()->addVariableValue(elem.getIdentifier(), getFirstValue(elem.getValue()));
    } else {
      ifResolverData.top()->addVariableValue(elem.getIdentifier(), elem.getValue());
    }
  } else if (valueIsKnown(elem.getValue())) {
    // update the variable's value in the map (-> without checking if this variable was declared before)
    variableValues[elem.getVarTargetIdentifier()] = getFirstValue(elem.getValue());
    // mark this statement for deletion as we don't need it anymore
    nodesQueuedForDeletion.push(&elem);
    // mark this statement as evaluated to notify its parent about a known value
    storeEvaluatedNode(&elem, getFirstValue(elem.getValue()));
  }
}

void CompileTimeExpressionSimplifier::handleBinaryExpressions(AbstractNode &elem,
                                                              AbstractExpr *leftOperand,
                                                              AbstractExpr *rightOperand) {
  auto leftValueIsKnown = valueIsKnown(leftOperand);
  auto rightValueIsKnown = valueIsKnown(rightOperand);
  if (leftValueIsKnown && rightValueIsKnown) {
    // if both operand values are known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&elem, getTransformedVariableMap());
    storeEvaluatedNode(&elem, std::vector<AbstractExpr *>(result.begin(), result.end()));
  } else if (leftValueIsKnown || rightValueIsKnown) {
    // if only one of both is known -> replace its values by the evaluated one
    auto concernedOperand = leftValueIsKnown ? leftOperand : rightOperand;
    auto newValue = getFirstValue(concernedOperand);
    elem.removeChildBilateral(concernedOperand);
    elem.addChild(newValue);
  }
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
  auto operandValueIsKnown = valueIsKnown(elem.getRight());
  if (operandValueIsKnown) {
    // if operand value is known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&elem, getTransformedVariableMap());
    storeEvaluatedNode(&elem, std::vector<AbstractExpr *>(result.begin(), result.end()));
  }
}

void CompileTimeExpressionSimplifier::visit(Block &elem) {
  Visitor::visit(elem);
  // If all Block's statements are marked for deletion, mark it as evaluated to notify its parent.
  // The parent is then responsible to decide whether it makes sense to delete this Block or not.
  // An example for a Block that cannot be deleted:
  //  computeSth(plaintext_int v) {
  //    ...
  //    if (v > 10) {     // <- this block cannot be deleted trivially (need to reverse criterion to flip branches)
  //      int a = 22;     // <- this statement can be deleted after saving a's value, then Block is empty
  //    } else {
  //      v = v*2;
  //    }
  //  }

  // check if there is any statement within this Block that is not marked for deletion
  bool allStatementsInBlockAreMarkedForDeletion = true;
  for (auto &statement : elem.getStatements()) {
    if (evaluatedNodes.count(statement)==0) {
      allStatementsInBlockAreMarkedForDeletion = false;
      break;
    }
//    if (nodesQueuedForDeletion.find(statement)==0) {
//      allStatementsInBlockAreMarkedForDeletion = false;
//      break;
//    }
  }
  // if all statements of this Block are marked for deletion, we can mark the Block as evaluable
  // -> let the Block's parent decide whether to keep this empty Block or not
  if (allStatementsInBlockAreMarkedForDeletion) {
    storeEvaluatedNode(&elem, nullptr);
  }
}

void CompileTimeExpressionSimplifier::visit(Variable &elem) {
  Visitor::visit(elem);
  // if the variable's value is known yet mark the node as evaluated
  if (valueIsKnown(&elem))
    storeEvaluatedNode(&elem, getFirstValue(&elem));
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
  if (std::all_of(parameters.begin(), parameters.end(), [this](FunctionParameter *fp) { return valueIsKnown(fp); })) {
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
}

void CompileTimeExpressionSimplifier::visit(FunctionParameter &elem) {
  Visitor::visit(elem);

  //  auto var = dynamic_cast<Variable *>(elem.getValue());
//  if (var!=nullptr) {
//    // Store the initial variable value, that is the value that was passed to the Function.
//    // For example: computeX(int x) would store ("x", new Variable("x")).
//    variableValues[var->getIdentifier()] = var;
//  }

// This simplifier does not care about the variable's datatype, hence we can mark this node as evaluable (= deletion
  // candidate). This mark is only relevant in case that this FunctionParameter is part of a Function referred by a Call
  // statement because Call statements can be replaced by inlining the Function's computation.
  storeEvaluatedNode(&elem, nullptr);
}

void CompileTimeExpressionSimplifier::visit(If &elem) {
  // We need to make sure that at the end of visiting the If statement's children, our variableValues map reflects the
  // latest values.

  // Bypass the base Visitor's logic and directly visit the condition only
  elem.getCondition()->accept(*this);

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
    if (thenAlwaysExecuted) {
      if (valueIsKnown(elem.getThenBranch()))
        wholeIfStmtCanBeDeleted = true;
    } else {
      if (elem.getElseBranch()==nullptr || valueIsKnown(elem.getElseBranch()))
        wholeIfStmtCanBeDeleted = true;
    }
    // enqueue the If statement and its children for deletion
    if (wholeIfStmtCanBeDeleted) nodesQueuedForDeletion.push(&elem);

  } else {
    // if we don't know the If statement's value -> resolve the If statement

    // set resolveIfStatementActive = true; b = [condition], notB = (1-[condition])
    ifResolverData.push(new IfStatementResolverData(elem.getCondition()));

    // visit the thenBranch
    elem.getThenBranch()->accept(*this);
    // if exists, visit the elseBranch
    if (elem.getElseBranch()!=nullptr) {
      ifResolverData.top()->setActiveBranch(false);
      elem.getElseBranch()->accept(*this);
    }
    // TODO(pjattke): modify visits of AST objects to consider resolveIfStatementActive

    //  Generate If-dependent assignment values by using map ifStatementVariableValues
    //  if either one of the pair's values (first, second) is empty, we need to pass the value's current value instead
    //  such that the value stays the same
    //  Consider following example where 22 is the variable's original value that needs to be preserved if not [a>42]:
    //    int i = 22; if (a > 42) { i = 111; }  --> int i = [a>42]*111 + [1-[a>42]]*22;
    auto currentIfResolverStruct = ifResolverData.top();
    for (auto &[varIdentifier, valueThenElse] : currentIfResolverStruct->getIfStatementVariableValues()) {
      //
      if (valueThenElse.first==nullptr || valueThenElse.second==nullptr) {
        // retrieve old value
        AbstractExpr *oldValue;
        if (variableValues.count(varIdentifier) > 0) {
          oldValue = variableValues.at(varIdentifier);
        } else {
          oldValue = new Variable(varIdentifier);
        }
        // save new value that depends on the If statement's condition
        if (valueThenElse.first==nullptr) {
          variableValues[varIdentifier] =
              ifResolverData.top()->generateIfDependentValue(oldValue, valueThenElse.second);
        } else if (valueThenElse.second==nullptr)
          variableValues[varIdentifier] = ifResolverData.top()->generateIfDependentValue(valueThenElse.first, oldValue);
      } else {

        variableValues[varIdentifier] =
            ifResolverData.top()->generateIfDependentValue(valueThenElse.first, valueThenElse.second);
      }
    }

    // set resolveIfStatementActive = false
    ifResolverData.pop();

    // TODO
    // move all non-evaluable statements to the If statement's parent node
//    if (elem.getParentsNonNull().size() > 1) {
//      throw std::logic_error("Unexpected number of parents (>1) of If statement!"
//                             "Cannot continue because it is unclear where to move non-evaluable children to.");
//    }
//    AbstractNode *ifStatementParent = elem.getParentsNonNull().front();
//    moveChildIfNotEvaluable(ifStatementParent, elem.getThenBranch());
//    if (elem.getElseBranch()!=nullptr) {
//      moveChildIfNotEvaluable(ifStatementParent, elem.getElseBranch());
//    }

    // enqueue the If statement for deletion
    nodesQueuedForDeletion.push(&elem);
  }
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
    return variableValues.count(abstractExprAsVariable->getIdentifier()) > 0;
//    // and the variable's value is not a symbolic one (i.e., not a Variable itself)
//    variableValueIsKnown =
//        valueExists && dynamic_cast<Variable *>(variableValues.at(abstractExprAsVariable->getIdentifier()))==nullptr;
  }
  // ii.) or the node was evaluated before (i.e., its value is in evaluatedNodes)
  return variableValueIsKnown || evaluatedNodes.count(abstractExpr) > 0;
}

void CompileTimeExpressionSimplifier::storeEvaluatedNode(
    AbstractNode *node, const std::vector<AbstractExpr *> &evaluationResult) {
  evaluatedNodes.emplace(node, evaluationResult);
}

void CompileTimeExpressionSimplifier::storeEvaluatedNode(
    AbstractNode *node, AbstractExpr *evaluationResult) {
  evaluatedNodes.insert(std::make_pair(node, std::vector<AbstractExpr *>({evaluationResult})));
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
//    if (auto varAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(variableValues.at(nodeAsVariable->getIdentifier())))
//      return varAsAbstractLiteral;
  }

  // in any other case -> search the given node in the map of already evaluated nodes
  auto it = evaluatedNodes.find(node);
  if (it==evaluatedNodes.end()) {
    // getFirstValue(...) assumes that the caller knows that the node was already evaluated (check using valueIsKnown
    // before calling it) -> throw an exception if no value was found.
    throw std::invalid_argument("Could not find any value for Node " + node->getUniqueNodeId() + ". ");
  } else if (it->second.size() > 1) {
    // if more than one evaluation results exists the user probably didn't know about that -> throw an exception
    throw std::logic_error(
        "Using getFirstValue(...) for expressions that return more than one value is probably undesired.");
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

void CompileTimeExpressionSimplifier::moveChildIfNotEvaluable(AbstractNode *ifStatementsParent,
                                                              AbstractStatement *branchStatement) {
  // if branchStatement's value is known -> nothing to do, return
  if (valueIsKnown(branchStatement)) return;

  // otherwise we need to move those statements to the ifStatementsParent that are not known
  for (auto &stmt : branchStatement->getChildrenNonNull()) {
    // if the current statement is not evaluable, then move it as child to If's parent
    if (!valueIsKnown(stmt)) {
      branchStatement->removeChildBilateral(stmt);
      ifStatementsParent->addChild(stmt, true);
    }
  }
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

AbstractLiteral *CompileTimeExpressionSimplifier::getDefaultVariableInitializationValue(Types datatype) {
  switch (datatype) {
    case Types::BOOL:return new LiteralBool("false");
    case Types::INT:return new LiteralInt(0);
    case Types::FLOAT:return new LiteralFloat(0.0f);
    case Types::STRING:return new LiteralString("");
    default:
      throw std::invalid_argument("Cannot determine the default value for the given AbstractLiteral!"
                                  "Given Literal is not a LiteralBool, LiteralFloat, LiteralInt, or LiteralString.");
  }
}
