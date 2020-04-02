#include <ControlFlowGraphVisitor.h>
#include "PrintVisitor.h"
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
#include "For.h"
#include "ParameterList.h"
#include "CallExternal.h"
#include "While.h"
#include "Call.h"
#include "Rotate.h"
#include "Transpose.h"
#include "OperatorExpr.h"

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

void CompileTimeExpressionSimplifier::visit(Rotate &elem) {
  Visitor::visit(elem);
  // if the Rotate's operand is known at compile-time, we can execute the rotation and replace this node by the
  // rotation's result (i.e., rotated operand)
  if (valueIsKnown(elem.getOperand()) && valueIsKnown(elem.getRotationFactor())) {
    auto val = getFirstValue(elem.getOperand());
    // we need a AbstractLiteral to be able to perform the rotation
    if (auto valAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(val)) {
      // clone the AbstractLiteral (including its value)
      auto clonedVal = valAsAbstractLiteral->clone(false)->castTo<AbstractLiteral>();
      // perform rotation on the cloned literal
      clonedVal->getMatrix()->rotate(getFirstValue(elem.getRotationFactor())->castTo<LiteralInt>()->getValue(), true);
      // replace this Rotate node by a new node containing the rotated operand
      elem.getOnlyParent()->replaceChild(&elem, clonedVal);
      nodesQueuedForDeletion.push_back(&elem);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(Transpose &elem) {
  Visitor::visit(elem);
  // if the Transpose' operand is known at compile-time, we can execute the transpose cmd and replace this node by the
  // transpose result (i.e., transposed operand)
  if (valueIsKnown(elem.getOperand())) {
    auto val = getFirstValue(elem.getOperand());
    // we need a AbstractLiteral to be able to perform the rotation
    if (auto valAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(val)) {
      // clone the AbstractLiteral (including its value)
      auto clonedVal = valAsAbstractLiteral->clone(false)->castTo<AbstractLiteral>();
      // perform transpose on the cloned literal
      clonedVal->getMatrix()->transpose(true);
      // replace this Rotate node by a new node containing the rotated operand
      elem.getOnlyParent()->replaceChild(&elem, clonedVal);
      nodesQueuedForDeletion.push_back(&elem);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(GetMatrixElement &elem) {
  Visitor::visit(elem);
  // if this is an expression like "matrix[a][b]" where the operand (matrix) as well as both indices (a,b) are known
  if (valueIsKnown(elem.getOperand()) && valueIsKnown(elem.getRowIndex()) && valueIsKnown(elem.getColumnIndex())) {
    // get the row index
    int rowIndex = getFirstValue(elem.getRowIndex())->castTo<LiteralInt>()->getValue();
    // get the column index
    int columnIndex = getFirstValue(elem.getColumnIndex())->castTo<LiteralInt>()->getValue();
    // get the element at position (row, column)
    auto matrix = dynamic_cast<AbstractLiteral *>(getFirstValue(elem.getOperand()))->getMatrix();
    auto retrievedElement = matrix->getElementAt(rowIndex, columnIndex);
    // replace this GetMatrixElement referred by the parent node by the retrieved element
    elem.getOnlyParent()->replaceChild(&elem, retrievedElement);
  }
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
    elem.deleteNode(&nodeToBeDeleted, true);
  }
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getRootNode());
}

void CompileTimeExpressionSimplifier::visit(Datatype &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(CallExternal &elem) {
  Visitor::visit(elem);
  cleanUpAfterStatementVisited(elem.AbstractExpr::castTo<AbstractExpr>(), false);
}

// =====================
// Simplifiable Statements
// =====================

template<typename T, typename U>
void simplifyAbstractExprMatrix(U &elem) {
  if (auto mx = dynamic_cast<Matrix<AbstractExpr *> *>(elem.getMatrix())) {
    auto mxDim = mx->getDimensions();
    std::vector<std::vector<T>> simplifiedVec(mxDim.numRows, std::vector<T>(mxDim.numColumns));
    for (int i = 0; i < mxDim.numRows; ++i) {
      for (int j = 0; j < mxDim.numColumns; ++j) {
        auto curElement = mx->getElementAt(i, j);
        if (auto elemAsBool = dynamic_cast<U *>(curElement)) {
          simplifiedVec[i][j] = elemAsBool->getValue();
        } else {
          // we cannot simplify this matrix as element (i,j) is of type non-T but Matrix<T> can only hold T vals
          return;
        }
      }
    }
    auto parent = elem.getOnlyParent();
    parent->replaceChild(&elem, new U(new Matrix<T>(simplifiedVec)));
  }
}

void CompileTimeExpressionSimplifier::visit(LiteralBool &elem) {
  Visitor::visit(elem);
  simplifyAbstractExprMatrix<bool, LiteralBool>(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralInt &elem) {
  Visitor::visit(elem);
  simplifyAbstractExprMatrix<int, LiteralInt>(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralString &elem) {
  Visitor::visit(elem);
  simplifyAbstractExprMatrix<std::string, LiteralString>(elem);
}

void CompileTimeExpressionSimplifier::visit(LiteralFloat &elem) {
  Visitor::visit(elem);
  simplifyAbstractExprMatrix<float, LiteralFloat>(elem);
}

void CompileTimeExpressionSimplifier::visit(Variable &elem) {
  Visitor::visit(elem);
  if (variableValues.count(elem.getIdentifier()) > 0 && replaceVariablesByValues) {
    // if we know the variable's value (i.e., its value is either any subtype of AbstractLiteral or an AbstractExpr if
    // this is a symbolic value that defines on other variables), we can replace this variable node by its value
    auto variableParent = elem.getOnlyParent();
    auto newValue = getFirstValue(&elem);
    variableParent->replaceChild(&elem, newValue);
  }
}

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  Visitor::visit(elem);
  // determine the variable's value
  AbstractExpr *variableValue;
  auto variableInitializer = elem.getInitializer();
  if (variableInitializer==nullptr) {
    variableValue = Datatype::getDefaultVariableInitializationValue(elem.getDatatype()->getType());
  } else {
    variableValue = variableInitializer;
  }
  // store the variable's value
  addVariableValue(elem.getVarTargetIdentifier(), variableValue);
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(variableInitializer);
  // mark this statement as removable as it is deleted
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::visit(VarAssignm &elem) {
  Visitor::visit(elem);
  // store the variable's value
  addVariableValue(elem.getVarTargetIdentifier(), elem.getValue());
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getValue());
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::visit(ArithmeticExpr &elem) {
  Visitor::visit(elem);
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getLeft());
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOperator());

  // transform this ArithmeticExpr into an OperatorExpr
  auto op = elem.getOperator();
  op->removeFromParents();
  std::vector<AbstractExpr *> operands{elem.getLeft(), elem.getRight()};
  elem.getLeft()->removeFromParents();
  elem.getRight()->removeFromParents();
  auto operatorExpr = new OperatorExpr(op, operands);
  elem.getOnlyParent()->replaceChild(&elem, operatorExpr);
  nodesQueuedForDeletion.push_back(&elem);
  operatorExpr->accept(*this);
}

void CompileTimeExpressionSimplifier::visit(LogicalExpr &elem) {
  Visitor::visit(elem);
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getLeft());
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOperator());

  // transform this LogicalExpr into an OperatorExpr
  auto op = elem.getOperator();
  op->removeFromParents();
  std::vector<AbstractExpr *> operands{elem.getLeft(), elem.getRight()};
  elem.getLeft()->removeFromParents();
  elem.getRight()->removeFromParents();
  auto operatorExpr = new OperatorExpr(op, operands);
  elem.getOnlyParent()->replaceChild(&elem, operatorExpr);
  nodesQueuedForDeletion.push_back(&elem);
  operatorExpr->accept(*this);
}

void CompileTimeExpressionSimplifier::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOperator());

  // check if the unary expression can be evaluated such that we can replace the whole node by its evaluation result
  if (dynamic_cast<AbstractLiteral *>(elem.getRight())) {
    // if operand value is known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&elem, getTransformedVariableMap());
    // replace this UnaryExpr by the evaluation's result
    elem.getOnlyParent()->replaceChild(&elem, result.front());
  } else {
    // if this UnaryExpr cannot be evaluated, replace the UnaryExpr by an OperatorExpr
    auto op = elem.getOperator();
    op->removeFromParents();
    std::vector<AbstractExpr *> operands{elem.getRight()};
    elem.getRight()->removeFromParents();
    auto operatorExpr = new OperatorExpr(op, operands);
    elem.getOnlyParent()->replaceChild(&elem, operatorExpr);
  }
  nodesQueuedForDeletion.push_back(&elem);
}

void CompileTimeExpressionSimplifier::visit(OperatorExpr &elem) {
  // Check if any of the operands is itself an OperatorExpr with the same symbol such that its operands can be merged
  // into this OperatorExpr. As we do not consider the operator's commutativity, we need to take care of the operands'
  // order while merging. For example:
  //
  //   Before merging:                After merging:
  //   ┌───┬───┬───┬───┬───┬───┐     ┌───┬───┬───┬───┬───┬───┬───┐
  //   │ + │ a │ │ │ b │ c │ d │     │ + │ a │ e │ f │ b │ c │ d │
  //   └───┴───┴─┼─┴───┴───┴───┘     └───┴───┴───┴───┴───┴───┴───┘
  //             │
  //       ┌───┬─▼─┬───┐
  //       │ + │ e │ f │
  //       └───┴───┴───┘
  std::vector<AbstractExpr *> newOperands;
  auto operatorAndOperands = elem.getChildren();
  for (auto it = operatorAndOperands.begin() + 1; it!=operatorAndOperands.end(); ++it) {
    auto operandAsOperatorExpr = dynamic_cast<OperatorExpr *>(*it);
    // check if this operand is an OperatorExpr of the same symbol
    if (operandAsOperatorExpr!=nullptr
        && operandAsOperatorExpr->getOperator()->equals(elem.getOperator()->getOperatorSymbol())) {
      auto operandsToBeAdded = operandAsOperatorExpr->getOperands();
      // go through all operands of this sub-OperatorExpr, remove each from its current parent and add it as operand to
      // this OperatorExpr
      for (auto &operand : operandsToBeAdded) {
        operand->removeFromParents();
        newOperands.push_back(operand->castTo<AbstractExpr>());
      }
      // mark the obsolete OperatorExpr child for deletion
      nodesQueuedForDeletion.push_back(*it);
    } else {
      // if this operand is not an OperatorExpr, we also need to remove it from this OperatorExpr because re-adding it
      // as operand would otherwise lead to having two times the same parent
      (*it)->removeFromParents();
      newOperands.push_back((*it)->castTo<AbstractExpr>());
    }
  }
  // replaced the operands by the merged list of operands also including those operands that were not an OperatorExpr
  auto curOperator = elem.getOperator();
  curOperator->removeFromParents();
  elem.setAttributes(curOperator, newOperands);

  // if this OperatorExpr is a logical expression, try to apply Boolean laws to further simplify this expression
  if (elem.isLogicalExpr()) simplifyLogicalExpr(elem);

  // If there is only one operand in this OperatorExpr left then we already applied the operator on all operands, hence
  // we can replace the whole OperatorExpr by its operand.
  // Exception: If this is an unary operator (e.g., !a) containing an unknown operand (e.g., Variable a) then this
  // replacement is not legal. This case is excluded by !elem.getOperator()->isUnaryOp().
  if (elem.getOperands().size()==1 && !elem.getOperator()->isUnaryOp()) {
    elem.getOnlyParent()->replaceChild(&elem, elem.getOperands().at(0));
  }
}

void CompileTimeExpressionSimplifier::simplifyLogicalExpr(OperatorExpr &elem) {
  // Simplifying this OperatorExpr using Boolean laws is only applicable if this OperatorExpr contains a logical
  // operator and there are at least two operands because we'll potentially remove one operand. If there exists already
  // only one operand, the operand must be moved up to the parent instead (not handled by simplifyLogicalExpr).
  if (!elem.isLogicalExpr() || elem.getOperands().size() <= 1) {
    return;
  }

  // collect all known operands from current OperatorExpr
  std::vector<AbstractLiteral *> knownOperands;
  for (auto &c : elem.getOperands()) {
    auto valueAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(c);
    if (valueAsAbstractLiteral!=nullptr && !valueAsAbstractLiteral->getMatrix()->containsAbstractExprs()) {
      knownOperands.push_back(valueAsAbstractLiteral);
    }
  }
  // The following logic requires that there is exactly one known operand. This should be the case if there are any
  // known operands because logical operators (AND, OR, XOR) are commutative and thus adding new operands always
  // performs aggregation. For example, adding (true XOR false XOR false XOR a) to OperatorExpr will partially evaluate
  // the expression and add (true XOR a) instead. Hence there should be at most one known operand, if there is any
  // known operand at all.
  if (knownOperands.size()!=1) return;

  // retrieve the known operand and the logical operator
  auto knownOperand = knownOperands.at(0);
  LogCompOp logicalOperator = std::get<LogCompOp>(elem.getOperator()->getOperatorSymbol());

  if (logicalOperator==LOGICAL_AND) {
    // - knownOperand == false: replace the whole expression by False as <anything> AND False is always False
    // - knownOperand == true: remove True from the expression as <anything> AND True only depends on <anything>
    if (knownOperand->isEqual(new LiteralBool(false))) {
      elem.getOnlyParent()->replaceChild(&elem, knownOperand);
    } else if (knownOperand->isEqual(new LiteralBool(true))) {
      elem.removeChild(knownOperand);
    }
  } else if (logicalOperator==LOGICAL_OR) {
    // - knownOperand == true: replace whole expression by True as <anything> OR True is always True
    // - knownOperand == false: remove False from the expression as <anything> OR False only depends on <anything>
    if (knownOperand->isEqual(new LiteralBool(true))) {
      elem.getOnlyParent()->replaceChild(&elem, knownOperand);
    } else if (knownOperand->isEqual(new LiteralBool(false))) {
      elem.removeChild(knownOperand);
    }
  } else if (logicalOperator==LOGICAL_XOR && knownOperand->isEqual(new LiteralBool(false))) {
    // - knownOperand == false: remove False from the expression as <anything> XOR False always depends on <False>
    // - knownOperand == true [not implemented]: rewrite <anything> XOR True to !<anything>
    elem.removeChild(knownOperand);
  }
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
    // clean up removableNodes result from children that indicates whether a child node can safely be deleted
    removableNodes.erase(statement);
  }
  // if all statements of this Block are marked for deletion, we can mark the Block as removable
  // -> let the Block's parent decide whether to keep this empty Block or not
  if (allStatementsInBlockAreMarkedForDeletion) markNodeAsRemovable(&elem);

  cleanUpAfterStatementVisited(&elem, false);
}

void CompileTimeExpressionSimplifier::visit(Call &elem) {
  Return *returnStmt = (elem.getFunc()!=nullptr) ? dynamic_cast<Return *>(elem.getFunc()->getBodyStatements().back())
                                                 : nullptr;
  // only perform inlining if...
  // there is a Return in the called function
  if (returnStmt!=nullptr) {
    auto returnStatementDescendants = returnStmt->getDescendants();
    if (// the Return statement does not have more than 20 descendant nodes ("threshold")
        returnStatementDescendants.size() <= 20
            // their is exactly one return value (because assignment of multiple values cannot be expressed yet
            && returnStmt->getReturnExpressions().size()==1) {
      // replace variables values of Call with those in called function
      auto parameterValues = elem.getParameterList()->getParameters();
      auto expectedFunctionParameters = elem.getFunc()->getParameterList()->getParameters();
      if (parameterValues.size()!=expectedFunctionParameters.size()) {
        throw std::invalid_argument("Number of given and expected parameters in Call does not match!");
      }
      // generate a map consisting of "variableIdentifier : variableValue" entries where variableIdentifier is the name
      // of the variable within the called function and variableValue the value (literal or variable) that is passed as
      // value for that identifier as part of the function call
      std::unordered_map<std::string, AbstractExpr *> varReplacementMap;
      for (int i = 0; i < parameterValues.size(); ++i) {
        auto variable = expectedFunctionParameters[i]->getValue()->castTo<Variable>();
        auto entry = std::make_pair(variable->getIdentifier(), parameterValues[i]->getValue());
        varReplacementMap.insert(entry);
      }
      for (auto &node : returnStatementDescendants) {
        // if the current node is a Variable node and it is a function parameter -> replace it
        auto nodeAsVariable = dynamic_cast<Variable *>(node);
        if (nodeAsVariable!=nullptr && varReplacementMap.count(nodeAsVariable->getIdentifier()) > 0) {
          node->getOnlyParent()->replaceChild(node,
                                              varReplacementMap.at(nodeAsVariable->getIdentifier())->clone(false));
        }
      }

      // remove return expression from its parent (return statement) and replace Call by extracted return statement
      auto parentNode = elem.getOnlyParent();
      auto returnExpr = returnStmt->getReturnExpressions().front();
      returnExpr->removeFromParents(true);
      parentNode->replaceChild(&elem, returnExpr);

      // revisit subtree as new simplification opportunities may exist now
      parentNode->accept(*this);
    }
  }
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::visit(ParameterList &elem) {
  Visitor::visit(elem);
  // if all of the FunctionParameter children are marked as removable, mark this node as removable too
  bool allFunctionParametersAreRemovable = true;
  for (auto &fp : elem.getParameters()) {
    if (!valueIsKnown(fp)) allFunctionParametersAreRemovable = false;
    // clean up removableNodes result from children that indicates whether a child node can safely be deleted
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
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
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
  if (dynamic_cast<LiteralBool *>(elem.getCondition())!=nullptr) {
    // If we know the condition's value, we can eliminate the branch that is never executed.
    // We need to detach this branch immediately, otherwise the variableValues map will only contain the values of the
    // last visited branch (but we need the values of the branch that is always executed instead).
    auto thenAlwaysExecuted = elem.getCondition()->castTo<LiteralBool>()->getValue();
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
      condition->removeFromParents();
      auto newCondition = new OperatorExpr(new Operator(UnaryOp::NEGATION), {condition});
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
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
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

std::unordered_set<std::string> CompileTimeExpressionSimplifier::removeVarsWrittenInStatementsFromVarValuesMap(
    Block &blockStmt) {
  // Erase any variable from variableValues that is writtenin the loop's body such that it is not replaced by any value
  // while visiting the body's statements. This includes the iteration variable as we moved the update statement into
  // the loop's body.
  // - create a new instance of the ControlFlowGraphVisitor
  ControlFlowGraphVisitor cfgv;
  // this is important as we're not using the AST as entry point, thus the scope logic won't work properly
  cfgv.setIgnoreScope(true);
  // - a set of variables that are written by the body's statements
  std::unordered_set<std::string> writtenVars;
  // - for each statement in the For-loop's body, collect all written variables
  for (auto &bodyStatement: blockStmt.getStatements()) {
    bodyStatement->accept(cfgv);
    auto writtenVarsInStatement = cfgv.getLastVariableWrites();
    writtenVars.insert(writtenVarsInStatement.begin(), writtenVarsInStatement.end());
  }
  // - remove all variables from the current variablesValues that are written in the For-loop's body
  for (auto &varIdentifier : writtenVars) variableValues.erase(varIdentifier);

  return writtenVars;
}

void CompileTimeExpressionSimplifier::visit(For &elem) {
  //TODO: IF compile time known & short enough, do full "symbolic eval/unroll" -> no for loop left

  // create a new Block statemeny: necessary for cleanup loop and to avoid overriding user-defined variables that expect
  // the initalizer to be within the for-loop's scope only
  auto blockEmbeddingLoops = new Block(); // ISSUE

  // move initializer from For-loop into newly added block
  auto forLoopInitializer = elem.getInitializer();
  forLoopInitializer->removeFromParents();
  blockEmbeddingLoops->addChild(forLoopInitializer);

  // wrap this For loop into the new block statement
  elem.getOnlyParent()->replaceChild(&elem, blockEmbeddingLoops);
  blockEmbeddingLoops->addChild(&elem);

  // create copies to allow reverting changes made by visiting nodes
  // - a copy of known variables with their values
  std::unordered_map<std::string, AbstractExpr *> variableValuesBackup = variableValues;
  // - a copy of all nodes marked for deletion
  auto nodesQueuedForDeletionCopy = nodesQueuedForDeletion;
  // - copies of the For-loop's condition, update statement, and body (required for cleanup loop)
  auto cleanupForLoop = elem.clone(false)->castTo<For>();
  auto ignrd =
      removeVarsWrittenInStatementsFromVarValuesMap(*cleanupForLoop->getStatementToBeExecuted()->castTo<Block>());
  cleanupForLoop->getCondition()->accept(*this);
  cleanupForLoop->getStatementToBeExecuted()->accept(*this);
  cleanupForLoop->getUpdateStatement()->accept(*this);
  // we need to use the for-loop's getter again because visiting the children may lead to replacement of nodes, e.g.,
  // ArithmeticExpr -> OperatorExpr
  auto forLoopCopyCondition = cleanupForLoop->getCondition();
  auto forLoopCopyBody = cleanupForLoop->getStatementToBeExecuted();
  auto forLoopCopyUpdateStmt = cleanupForLoop->getUpdateStatement();
  variableValues = variableValuesBackup;
  nodesQueuedForDeletion = nodesQueuedForDeletionCopy;

  // visit the intializer
  forLoopInitializer->accept(*this);
  auto variableValuesAfterVisitingInitializer = variableValues;
  // determine the loop variables, i.e., variables changed in the initializer statement
  auto loopVariablesMap = getChangedVariables(variableValuesBackup);

  // the new for-loop body containing the unrolled statements
  auto unrolledForLoopBody = new Block();
  // generate the unrolled loop statements by performing for each statement in the original's loop body:
  // duplicate the statement, replace all iteration variables, append the statement to the new (unrolled) loop body
  const int NUM_CIPHERTEXT_SLOTS = 3;
  for (int i = 0; i < NUM_CIPHERTEXT_SLOTS; i++) {
    // for each statement in the for-loop's body
    for (auto &stmt : elem.getStatementToBeExecuted()->castTo<Block>()->getStatements()) {
      // clone the body statement and append the statement to the unrolledForLoop Body
      unrolledForLoopBody->addChild(stmt->clone(false));
    }
    // add a copy of the update statement, visiting the body then automatically handles the iteration variable in the
    // cloned loop body statements - no need to manually adapt them
    unrolledForLoopBody->addChild(elem.getUpdateStatement()->clone(false));
  }
  // replace the for loop's body by the unrolled statements
  elem.replaceChild(elem.getStatementToBeExecuted(), unrolledForLoopBody);

  // delete update statement from loop since it's now incorporated into the body but keep a copy since we need it
  // for the cleanup loop
  elem.getUpdateStatement()->removeFromParents();

  // Erase any variable from variableValues that is writtenin the loop's body such that it is not replaced by any value
  // while visiting the body's statements. This includes the iteration variable as we moved the update statement into
  // the loop's body.
  auto writtenVars = removeVarsWrittenInStatementsFromVarValuesMap(*elem.getStatementToBeExecuted()->castTo<Block>());

  // restore the copy, otherwise the initializer visited after creating this copy would be marked for deletion
  nodesQueuedForDeletion = nodesQueuedForDeletionCopy;

  // make sure that variable declarations of variables written in the for-loop's are not marked for deletion because
  // they must be declared before using them
  for (auto it = nodesQueuedForDeletion.begin(); it!=nodesQueuedForDeletion.end();) {
    if (auto nodeAsVarDecl = dynamic_cast<VarDecl *>(*it)) {
      if (writtenVars.count(nodeAsVarDecl->getVarTargetIdentifier()) > 0) {
        it = nodesQueuedForDeletion.erase(it);
      } else {
        ++it;
      }
    }
  }

  // visit the for-loop's body to do inlining
  auto variableValuesBeforeVisitingLoopBody = variableValues;
  elem.getStatementToBeExecuted()->accept(*this);

  // visit the condition to replace loop iteration variables by the symbolic value of the variable that is last written
  // in the body, for example: i < 6 => i+3 < 6
  auto nodesQueuedForDeletionBeforeVisitingCondition = nodesQueuedForDeletion;
  elem.getCondition()->accept(*this);
  nodesQueuedForDeletion = nodesQueuedForDeletionBeforeVisitingCondition;

  // TODO: Future work:  Make this entire thing flexible with regard to num_slots_in_ctxt, i.e., allow changing how long
  //  unrolled loops are. Idea: generate all loops (see below cleanup loop ideas) starting from ludacriously large
  //  number, later disable/delete the ones that are larger than actually selected cipheretxt size determined from
  //  parameters?
  //  ISSUE: Parameters might depend on loop length? <-- This is a general issue (no loop bound => no bounded depth)

  // find all variables that were changed in the for-loop's body - even iteration vars (important for condition!) - and
  // emit them, i.e, create a new variable assignment for the variable
  auto bodyChangedVariables = getChangedVariables(variableValuesBeforeVisitingLoopBody);
  for (auto &[varIdentifier, varExpr] : bodyChangedVariables) {
    // create a new (only one!) statement for each variable changed in the body, this implicitly is the statement
    // with the "highest possible degree" of inlining for this variable
    elem.getStatementToBeExecuted()->castTo<Block>()->addChild(new VarAssignm(varIdentifier, varExpr));
    // remove all the variables that were changed within the body from variableValues as inlining them in any statement
    // after/outside the loop does not make any sense
    variableValues.erase(varIdentifier);
  }

  // TODO: Future work (maybe): for large enough num_..., e.g. 128 or 256, it might make sense to have a binary series
  //  of cleanup loops, e.g., if 127 iterations are left, go to 64-unrolled loop, 32-unrolled loop, etc.
  //  When to cut off? -> empirical decision?

  // handle remaining iterations using a "cleanup loop": place statements in a separate loop for remaining iterations
  // and attach the generated cleanup loop to the newly added Block
  // the cleanup loop consists of:
  // - initializer: reused from the unrolled loop as the loop variable is declared out of the first for-loop
  // - condition, update statement, body statements: remain the same as for the original loop
  blockEmbeddingLoops->addChild(cleanupForLoop);

  cleanUpAfterStatementVisited(reinterpret_cast<AbstractNode *>(&elem), false);
}

void CompileTimeExpressionSimplifier::visit(Return &elem) {
  Visitor::visit(elem);
  // simplify return expression: replace each evaluated expression by its evaluation result
  bool allValuesAreKnown = true;
  for (auto &returnExpr : elem.getReturnExpressions()) {
    if (!valueIsKnown(returnExpr)) allValuesAreKnown = false;
    // clean up removableNodes result from children that indicates whether a child node can safely be deleted
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
  // if node is a Literal: return the node itself
  if (auto nodeAsLiteral = dynamic_cast<AbstractLiteral *>(node)) {
    return nodeAsLiteral;
  }
  // if node is a variable: search the variable's value in the map of known variable values
  auto nodeAsVariable = dynamic_cast<Variable *>(node);
  if (nodeAsVariable!=nullptr && variableValues.count(nodeAsVariable->getIdentifier()) > 0) {
    // return a clone of te variable's value
    AbstractNode *pNode = variableValues.at(nodeAsVariable->getIdentifier())->clone(false);
    return dynamic_cast<AbstractExpr *>(pNode);
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

AbstractExpr *CompileTimeExpressionSimplifier::generateIfDependentValue(
    AbstractExpr *condition, AbstractExpr *trueValue, AbstractExpr *falseValue) {
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
    auto factorIsFalse = new OperatorExpr(new Operator(ArithmeticOp::SUBTRACTION),
                                          {new LiteralInt(1), condition->clone(false)->castTo<AbstractExpr>()});
    // case: trueValue == 0 && falseValue != 0 => value is 0 if the condition is True
    // -> return (1-b)*falseValue
    return new OperatorExpr(new Operator(MULTIPLICATION), {factorIsFalse, falseValue});
  } else if (falseValueIsNull) {
    // factorIsTrue = ifStatementCondition
    auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
    // case: trueValue != 0 && falseValue == 0 => value is 0 if the condition is False
    // -> return condition * trueValue
    return new OperatorExpr(new Operator(ArithmeticOp::MULTIPLICATION), {factorIsTrue, trueValue});
  }

  // default case: trueValue != 0 && falseValue != 0 => value is changed in both branches of If statement
  // -> return condition*trueValue + (1-b)*falseValue.
  // factorIsTrue = ifStatementCondition
  auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
  // factorIsFalse = [1-ifStatementCondition]
  auto factorIsFalse = new OperatorExpr(new Operator(ArithmeticOp::SUBTRACTION),
                                        {new LiteralInt(1), condition->clone(false)->castTo<AbstractExpr>()});
  return new OperatorExpr(
      new Operator(ArithmeticOp::ADDITION),
      {new OperatorExpr(new Operator(ArithmeticOp::MULTIPLICATION),
                        {factorIsTrue, trueValue->clone(false)->castTo<AbstractExpr>()}),
       new OperatorExpr(new Operator(ArithmeticOp::MULTIPLICATION),
                        {factorIsFalse, falseValue->clone(false)->castTo<AbstractExpr>()})});
}

void CompileTimeExpressionSimplifier::cleanUpAfterStatementVisited(
    AbstractNode *statement, bool enqueueStatementForDeletion) {
  // mark this statement for deletion as we don't need it anymore
  if (enqueueStatementForDeletion) nodesQueuedForDeletion.push_back(statement);
}

void CompileTimeExpressionSimplifier::addVariableValue(const std::string &variableIdentifier,
                                                       AbstractExpr *valueAnyLiteralOrAbstractExpr) {
  auto clonedVariableValue = valueAnyLiteralOrAbstractExpr->clone(false);
  clonedVariableValue->removeFromParents();
  variableValues[variableIdentifier] = clonedVariableValue->castTo<AbstractExpr>();
}

bool CompileTimeExpressionSimplifier::isQueuedForDeletion(const AbstractNode *node) {
  return std::find(nodesQueuedForDeletion.begin(), nodesQueuedForDeletion.end(), node)
      !=nodesQueuedForDeletion.end();
}

std::unordered_map<std::string, AbstractExpr *>
CompileTimeExpressionSimplifier::getChangedVariables(
    std::unordered_map<std::string, AbstractExpr *> variableValuesBeforeVisitingNode) {
  // the result list of changed variables with their respective value
  std::unordered_map<std::string, AbstractExpr *> changedVariables;
  // Loop through all variables in the current variableValues and check for each if it changed.
  // It is important that we loop through variableValues instead of variableValuesBeforeVisitingNode because there may
  // be newly declared variables.
  for (auto &[varIdentifier, expr] : variableValues) {
    // a variable is changed if it either was added (i.e., declaration of a new variable) or its value was changed
    if (variableValuesBeforeVisitingNode.count(varIdentifier)==0
        || (variableValuesBeforeVisitingNode.count(varIdentifier) > 0
            && !variableValuesBeforeVisitingNode.at(varIdentifier)->isEqual(expr))) {
      changedVariables.emplace(varIdentifier, expr);
    }
  }
  return changedVariables;
}

void CompileTimeExpressionSimplifier::visit(AbstractMatrix &elem) {
  Visitor::visit(elem);
}
