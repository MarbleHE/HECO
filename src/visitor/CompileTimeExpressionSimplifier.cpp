#include <typeinfo>
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
  if (evaluateExpressions &&
      valueIsKnown(elem.getOperand()) && valueIsKnown(elem.getRowIndex()) && valueIsKnown(elem.getColumnIndex())) {
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
    if (nodesAlreadyDeleted.count(nodeToBeDeleted) > 0) {
      throw std::runtime_error("ERROR: Trying to delete node twice. "
                               "Probably the node was by mistake enqueued multiple times for deletion.");
    }
    nodesAlreadyDeleted.insert(nodeToBeDeleted);
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
        } else { // we cannot simplify this matrix as element (i,j) is of type non-T but Matrix<T> can only hold T vals
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
    auto variableParent = elem.getOnlyParent();  // TODO replace newValue by getFirstValue(&elem)
    auto newValue = variableValues.at(elem.getIdentifier())->clone(false)->castTo<AbstractExpr>();
    variableParent->replaceChild(&elem, newValue);
    // if this newValue is an AbstractBinaryExpr by itself
    if (auto bexp = dynamic_cast<AbstractBinaryExpr *>(newValue)) {
      // Visit replaced subtree to consider its operands by binary expression accumulation. It is important to set
      // before replaceVariablesByValues=false, otherwise we potentially end up in an infinite recursion.
      // We can safely set replaceVariablesByValues=false as the variable replaced by this subtree is already the "best
      // possible" symbolic representation of each variable.
      replaceVariablesByValues = false;
      bexp->accept(*this);
      replaceVariablesByValues = true;
      //binaryExpressionAccumulator.addAllOperandsOfSubtree(bexp);
      //binaryExpressionAccumulator.setLastVisitedSubtree(bexp);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  binaryExpressionAccumulator.isVisitingVariableDeclarationOrAssignment = true;
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
  binaryExpressionAccumulator.isVisitingVariableDeclarationOrAssignment = true;
  Visitor::visit(elem);
  // store the variable's value
  addVariableValue(elem.getVarTargetIdentifier(), elem.getValue());
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getValue());
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::handleBinaryExpression(AbstractBinaryExpr &arithmeticExpr) {
  if (!doBinaryExpressionAccumulation) return;

  // left-hand side operand
  auto lhsOperand = arithmeticExpr.getLeft();
  auto leftValueIsKnown = valueIsKnown(lhsOperand);
  // right-hand side operand
  auto rhsOperand = arithmeticExpr.getRight();
  auto rightValueIsKnown = valueIsKnown(rhsOperand);

  if (leftValueIsKnown && rightValueIsKnown) {
    // if both operand values are known -> evaluate the expression and store its result
    auto result = evaluateNodeRecursive(&arithmeticExpr, getTransformedVariableMap());
    if (result.size() > 1) throw std::logic_error("Unexpected: Evaluation result contains more than one value.");
    arithmeticExpr.getOnlyParent()->replaceChild(&arithmeticExpr, result.front());
    nodesQueuedForDeletion.push_back(&arithmeticExpr);
    // stop execution here as there's nothing to collect in the binaryExpressionAccumulator
    return;
  }
  // update accumulator
  auto currentOperator = arithmeticExpr.getOperator();

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
    // if operator is non-associative and non-commutative -> return from function as operator is unsupported
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
    // if the parent is a OperatorExpr of the same operator, we can simply append the operands to it
    auto parentAsOperatorExpr = dynamic_cast<OperatorExpr *>(arithmeticExpr.getOnlyParent());
    if (parentAsOperatorExpr!=nullptr
        && parentAsOperatorExpr->getOperator()->equals(binaryExpressionAccumulator.operatorSymbol)) {
      for (auto &accumulatedOperand : binaryExpressionAccumulator.operands) {
        parentAsOperatorExpr->addOperand(accumulatedOperand);
      }
    } else {
      // ...otherwise we need to generate a new expression consisting of the operands collected so far
      auto treeRoot = binaryExpressionAccumulator.getSimplifiedSubtree();
      // replace the current arithmetic expression by the one generated out of the accumulated & simplified operands
      arithmeticExpr.getOnlyParent()->replaceChild(arithmeticExpr.castTo<AbstractNode>(), treeRoot);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(ArithmeticExpr &elem) {
  // continue traversal: visiting the expression's operands and operator
  Visitor::visit(elem);
  handleBinaryExpression(elem);
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getLeft());
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOperator());
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
    if ((elem.getOperator()->equals(LOGICAL_AND) && booleanOperand->getValue())
        || (elem.getOperator()->equals(LOGICAL_OR) && !booleanOperand->getValue())
        || (elem.getOperator()->equals(LOGICAL_XOR) && !booleanOperand->getValue())) {
      nonBooleanOperand->removeFromParents();
      elem.getOnlyParent()->replaceChild(&elem, nonBooleanOperand);
      nodesQueuedForDeletion.push_back(&elem);
    } else if (elem.getOperator()->equals(LOGICAL_AND) && !booleanOperand->getValue()) {
      // <anything> AND false  ==  false
      elem.getOnlyParent()->replaceChild(&elem, new LiteralBool(false));
      nodesQueuedForDeletion.push_back(nonBooleanOperand);
    } else if (elem.getOperator()->equals(LOGICAL_OR) && booleanOperand->getValue()) {
      // <anything> OR true  ==  true
      elem.getOnlyParent()->replaceChild(&elem, new LiteralBool(true));
      nodesQueuedForDeletion.push_back(nonBooleanOperand);
    } else if (elem.getOperator()->equals(LOGICAL_XOR)
        && booleanOperand->getValue()) {
      // <anything> XOR true  ==  NOT <anything>
      nonBooleanOperand->removeFromParents();
      auto uexp = new UnaryExpr(NEGATION, nonBooleanOperand);
      elem.getOnlyParent()->replaceChild(&elem, uexp);
      nodesQueuedForDeletion.push_back(&elem);
    }
  } else {
    // handles the case where both operands are known or neither one and tries to simplify nested logical expressions
    handleBinaryExpression(elem);
  }
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getLeft());
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOperator());
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
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getRight());
  removableNodes.erase(elem.getOp());
}

void CompileTimeExpressionSimplifier::visit(OperatorExpr &elem) {
  // collect known operands from current OperatorExpr and remove them from the AST
  std::vector<AbstractLiteral *> knownOperands;
  for (auto &c : elem.getOperands()) {
    if (!valueIsKnown(c)) continue;
    auto value = getFirstValue(c);
    if (auto valueAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(value)) {
      knownOperands.push_back(valueAsAbstractLiteral);
      elem.removeChild(c);
    }
  }

  // evaluate the operator on the known operands: result is a single AbstractLiteral
  auto evalResult = elem.getOperator()->applyOperator(knownOperands);

  if (elem.getOperands().size()==0) {
    // if there are no other (unknown) operands left, attach the evaluation result to the OperatorExpr's parent
    elem.getOnlyParent()->replaceChild(&elem, evalResult);
    nodesQueuedForDeletion.push_back(&elem);
  } else if (elem.isArithmeticExpr()) {
    elem.addOperand(evalResult);
  } else if (elem.isLogicalExpr()) {  // LogicalExpr implies that return value is a Bool
    LogCompOp elemOperator = std::get<LogCompOp>(elem.getOperator()->getOperatorSymbol());
    if (elemOperator==LOGICAL_AND) {
      // if the evalResult is false: replace the whole expression by false as <anything> AND false is always false
      // if the evalResult is true:
      // - if #remainingOperands<2: append partial evaluation result
      // - else: do nothing as expression's value only depends on remaining operands
      if (evalResult->isEqual(new LiteralBool(false))) {
        elem.getOnlyParent()->replaceChild(&elem, evalResult);
      } else if (evalResult->isEqual(new LiteralBool(true)) && elem.getOperands().size() < 2) {
        elem.addOperand(evalResult);
      }
    } else if (elemOperator==LOGICAL_OR) {
      // if the evalResult is true: replace whole expression by true as <anything> OR true is always true
      // if the evalResult is false:
      // - if #remainingOperands<2: append evalResult such that there are at least two operands
      // - else: do nothing as expression's value only depends on remaining operands
      if (evalResult->isEqual(new LiteralBool(true))) {
        elem.getOnlyParent()->replaceChild(&elem, evalResult);
      } else if (evalResult->isEqual(new LiteralBool(false)) && elem.getOperands().size() < 2) {
        elem.addOperand(evalResult);
      }
    } else if (elemOperator==LOGICAL_XOR
        && (evalResult->isEqual(new LiteralBool(true))
            || (evalResult->isEqual(new LiteralBool(false)) && elem.getOperands().size() < 2))) {
      // if the evalResult is true: add it to the remaining operands because it influences the result
      // if the evalResult is false:
      // - if #remainingOperands<2: append it to the remaining operands such that there are at least two operands
      // - else: do nothing as expression's value only depends on remaining operands
      elem.addOperand(evalResult);
    }
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
  if (returnStmt!=nullptr) {
    // only perform inlining if...
    // there is a Return in the called function
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
      auto newCondition = new UnaryExpr(UnaryOp::NEGATION, condition);
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

void CompileTimeExpressionSimplifier::visit(For &elem) {
  // save a copy of all variables and their values
  std::unordered_map<std::string, AbstractExpr *> variableValuesBackup = variableValues;

  // save a copy of all nodes marked for deletion
  // we'll use that copy later to "undo" marking nodes for deletion as we don't want to delete any nodes within that For
  auto nodesQueuedForDeletionCopy = nodesQueuedForDeletion;

  // visit the intializer
  auto initializerCopy = elem.getInitializer()->clone(true);
  elem.getInitializer()->accept(*this);
  // visit the condition, before clone the condition as visiting it will perform inlining and variables will be replaced
  // by values but we need to revisit and reevaluate the statement after executing the update statement
  auto conditionCopy = elem.getCondition()->clone(true);
  elem.getCondition()->accept(*this);
  // now we're able to determine whether the bounds are compile-time known

  auto variableValuesAfterVisitingInitializerAndConditionOnce = variableValues;

  // conditions that must be satisfied (and assumptions made) for loop unrolling
  // (using castTo following because dynamic_cast somehow doesn't work here...):
  // 1. For-loop's bounds are compile-time known
  bool boundsAreCompileTimeKnown = valueIsKnown(elem.getCondition());
  // 2. For-loop's initializer contains a single variable declaration (VarDecl)
  bool initializerContainsVarDecl = true;
  try { elem.getInitializer()->castTo<VarDecl>(); } catch (std::exception &e) { initializerContainsVarDecl = false; }
  // 3. For-loop's update statement contains a single variable assignment (VarAssignm)
  bool updaterContainsVarAssignm = true;
  try { elem.getUpdateStatement()->castTo<VarAssignm>(); } catch (std::exception &e) {
    updaterContainsVarAssignm = false;
  }

  // if the For-loop's bounds are compile-time known
  if (boundsAreCompileTimeKnown && initializerContainsVarDecl && updaterContainsVarAssignm) {
    // a vector containing maps of (variableIdentifier, variableValue) values; a map for each iteration as the loop
    // might multiple iteration variables (e.g., nested in a Block)
    std::vector<std::unordered_map<std::string, AbstractExpr *>> loopVariableValues;
    std::vector<AbstractExpr *> updateExpressions;

    // a helper function to evaluate the for loop's condition
    auto conditionIsTrue = [&]() {
      auto evaluatedCond = evaluateNodeRecursive(conditionCopy->castTo<AbstractNode>(), getTransformedVariableMap());
      return *evaluatedCond.front()==LiteralBool(true);
    };

    // create a backup (clone) of the original, non-inlined update statement
    auto updateStmtCopy = elem.getUpdateStatement()->clone(true);

    // Save all loop iteration variables with their respective value by comparing the map with the backup made before
    // visiting the loop's initializer. This is required to determine how often this loop will run and to determine the
    // loop variable values as long as the condition evaluates to True.
    while (conditionIsTrue()) {
      // a map of (variableIdentifier, variableValue) pairs for the current loop iteration
      std::unordered_map<std::string, AbstractExpr *> changedVariables;

      // find all changed variables
      for (auto &[varIdentifier, expr] : variableValues) {
        // a variable is changed if it either was added (i.e., declaration of a new variable) or its value was changed
        if (variableValuesBackup.count(varIdentifier)==0
            || (variableValuesBackup.count(varIdentifier) > 0 && variableValuesBackup.at(varIdentifier)!=expr)) {
          changedVariables.emplace(varIdentifier, expr);
        }
      }
      // store all modified variables: those are the loop iteration variables
      loopVariableValues.push_back(changedVariables);

      // visit the loop's update statement, we do this on a copy because visiting it will perform inlining (replace
      // variables by their value)
      updateStmtCopy->clone(false)->accept(*this);
    } // end of while

    // remove iteration variables from map, otherwise visiting the update statement will perform inlining
    for (auto &[varIdentifier, unused] : loopVariableValues.at(0)) variableValues.erase(varIdentifier);

    // a vector of symbolic loop iteration variables, e.g., i, i+1, i+2
    std::vector<AbstractExpr *> symbolicLoopIterationVars;

    // determine the loop's iteration variable (e.g., i)
    // assumption: initializer contains a single variable declaration statement
    auto loopIterationVariableIdentifier = initializerCopy->castTo<VarDecl>()->getVarTargetIdentifier();
    symbolicLoopIterationVars.push_back(new Variable(loopIterationVariableIdentifier));

    // generate symbolic iteration variables (e.g., i, i+1, i+2)
    for (int i = 0; i < loopVariableValues.size(); ++i) {
      auto tempClonedUpdateStmt = updateStmtCopy->clone(false);
      // visit the  update statement
      tempClonedUpdateStmt->accept(*this);
      // assumption: for-update contains exactly one variable assignment
      // do not add the last loop iteration variable as this is the one that does not satisfy the loop condition anymore
      if (i < loopVariableValues.size() - 1) {
        auto symbolicExpr = tempClonedUpdateStmt->castTo<VarAssignm>()->getValue();
        symbolicExpr->removeFromParents();
        symbolicLoopIterationVars.push_back(symbolicExpr);
      }
    }

    // a helper function that replaces all variables identified by the given loopVariableIdentifier by the expression
    // given as replacementTarget in the tree rooted in node subtreeRoot
    auto replaceVariables =
        [](AbstractNode *subtreeRoot, std::string loopVariableIdentifier, AbstractExpr *replacementTarget) {
          for (auto &n : subtreeRoot->getDescendants()) {
            auto variable = dynamic_cast<Variable *>(n);
            // only replace a node if it is a variable and the variable identifier matches the given one
            if (variable!=nullptr && variable->getIdentifier()==loopVariableIdentifier) {
              variable->getOnlyParent()->replaceChild(variable, replacementTarget);
            }
          }
        };

    // compute parameters for partial loop unrolling, e.g.,
    //  - newNumIterations = #iterations / 128
    //  - remainingIterationsCleanupLoop = #iterations % 128
    const int NUM_CIPHERTEXT_SLOTS = 3;
    int numNewIterations = loopVariableValues.size()/NUM_CIPHERTEXT_SLOTS;
    int numRemainingIterations = loopVariableValues.size()%NUM_CIPHERTEXT_SLOTS;

    // the new for-loop body containing the unrolled statements
    Block *unrolledForLoopBody = new Block();
    // generate the unrolled loop statements by performing for each statement in the original's loop body:
    // duplicate the statement, replace all iteration variables, append the statement to the new (unrolled) loop body
    for (int i = 0; i < NUM_CIPHERTEXT_SLOTS; i++) {
      // for each statement in the for-loop's body
      for (auto &stmt : elem.getStatementToBeExecuted()->castTo<Block>()->getStatements()) {
        // clone the body statements
        auto clonedStmt = stmt->clone(false);
        // replace the iteration variable by the one of the symbolicLoopIterationVars map
        replaceVariables(clonedStmt, loopIterationVariableIdentifier, symbolicLoopIterationVars.at(i));
        // append the statement to the unrolledForLoop Body
        unrolledForLoopBody->addChild(clonedStmt);
      }
    }
    // replace the for loop's body by the unrolled statements
    auto forLoopBody = elem.getStatementToBeExecuted();
    elem.replaceChild(elem.getStatementToBeExecuted(), unrolledForLoopBody);

    // make sure that the generated AST is valid
    CompileTimeExpressionSimplifier::validateAst(&elem);

    // for each variable: find the statement that last assigned the variable's value
    std::unordered_map<std::string, AbstractStatement *> lastAssignedVarValue;
    for (auto &stmt : elem.getStatementToBeExecuted()->getChildren()) {
      if (auto stmtAsVarAssignm = dynamic_cast<VarAssignm *>(stmt)) {
        lastAssignedVarValue[stmtAsVarAssignm->getVarTargetIdentifier()] = stmtAsVarAssignm;
      } else if (auto stmtAsVarDecl = dynamic_cast<VarDecl *>(stmt)) {
        lastAssignedVarValue[stmtAsVarDecl->getVarTargetIdentifier()] = stmtAsVarDecl;
      }
    }

    // pass 1: visit the for-loop's body to do inlining
    // erase the loop iteration variable (e.g., i) such that it is not replaced by its value
    variableValuesAfterVisitingInitializerAndConditionOnce.erase(loopIterationVariableIdentifier);
    variableValues = variableValuesAfterVisitingInitializerAndConditionOnce;
    nodesQueuedForDeletion = nodesQueuedForDeletionCopy;
    evaluateExpressions = false;
    doBinaryExpressionAccumulation = false;
    elem.getStatementToBeExecuted()->accept(*this);
    auto nodesQueuedForDeletionAfterFirstPass = nodesQueuedForDeletion;
    // pass 2: visit the for-loop's body to aggregate arithmetic/logical exprs into OperatorExprs
    // we need a second pass over the loop's body statements to aggregate remaining arithmetic/logical expressions into
    // OperatorExprs, this cannot be done in one step because if we force transformation of arithmetic/logical exprs
    // into OperatorExprs using alwaysGenerateSimplifiedSubtreeOfOperatorExpr, inlining would be performed using these
    // but afterward we could not simplify the "last statement in the loop" because binaryExpressionAccumulator does not
    // support OperatorExprs.

    goto skip;
    // TODO make OperatorExprs work with binaryExpressionAcc (at least for the ops supported by binaryExpressionAcc)
    variableValues = variableValuesAfterVisitingInitializerAndConditionOnce;
    nodesQueuedForDeletion = nodesQueuedForDeletionCopy;
    binaryExpressionAccumulator.alwaysGenerateSimplifiedSubtreeOfOperatorExpr = true;
    doBinaryExpressionAccumulation = true;
    elem.getStatementToBeExecuted()->accept(*this);
    skip:
    evaluateExpressions = true;
    doBinaryExpressionAccumulation = true;
    nodesQueuedForDeletion = nodesQueuedForDeletionAfterFirstPass;

    // TODO remove all variables from variableValues that were modified in the loop (e.g., sum)

    // remove statements that last assigned a variable's value from the nodesQueuedForDeletion vector
    // TODO what about variables declared within the loop's scope?
    for (auto &[varIdentifier, stmt] : lastAssignedVarValue) {
      // this is very inefficient!
      auto it = std::find(nodesQueuedForDeletion.begin(), nodesQueuedForDeletion.end(), stmt);
      if (it!=nodesQueuedForDeletion.end()) nodesQueuedForDeletion.erase(it);
    }

    // save the changed nodesQueuedForDeletion as our updated copy
    nodesQueuedForDeletionCopy = nodesQueuedForDeletion;

    // update the for loop's update statement (e.g., i=i+1 -> i=i+3)
    // assumption: update statement consists of single variable assignment only
    auto updateStmt = elem.getUpdateStatement()->castTo<VarAssignm>();
    updateStmt->replaceChild(updateStmt->getValue(), symbolicLoopIterationVars.at(NUM_CIPHERTEXT_SLOTS));

    // handle remaining iterations (cleanup loop) if remainingIterationsCleanupLoop != 0:
    // place statements in a separate loop for the remaining iterations
    if (numRemainingIterations!=0) {
      // build the new initializer expression, e.g., int i=6;
      auto initExpr = loopVariableValues.at(numNewIterations*NUM_CIPHERTEXT_SLOTS).at(loopIterationVariableIdentifier);
      auto clonedInitializer = elem.getInitializer()->clone(false)->castTo<VarDecl>();
      clonedInitializer->setAttributes(clonedInitializer->getIdentifier(), clonedInitializer->getDatatype(), initExpr);
      // the loop's condition (remains same), e.g., i < 7
      auto condition = conditionCopy;
      // the loop's update statement (remains same), e.g., i=i+1;
      auto updateStatement = updateStmtCopy->clone(false)->castTo<AbstractStatement>();
      // the loop's body statements (nested in a block; remains same)
      auto body = forLoopBody->clone(false)->castTo<AbstractStatement>();
      // attach the newly generated For "cleanup" loop to the function's body
      elem.getOnlyParent()->addChild(new For(clonedInitializer, condition, updateStatement, body));
    }
  } // end of: if (valueIsKnown(elem.getCondition))

  // restore the nodes that were queued for deletion before visiting this For handler; this is needed because otherwise
  // parts of the loop (e.g., VarDecl in initializer) will be deleted at the end of this visitor traversal
  nodesQueuedForDeletion = nodesQueuedForDeletionCopy;

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
                                            ArithmeticOp::SUBTRACTION,
                                            condition->clone(false)->castTo<AbstractExpr>());
    // case: trueValue == 0 && falseValue != 0 => value is 0 if the condition is True
    // -> return (1-b)*falseValue
    return new ArithmeticExpr(factorIsFalse, ArithmeticOp::MULTIPLICATION, falseValue);
  } else if (falseValueIsNull) {
    // factorIsTrue = ifStatementCondition
    auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
    // case: trueValue != 0 && falseValue == 0 => value is 0 if the condition is False
    // -> return condition * trueValue
    return new ArithmeticExpr(factorIsTrue, ArithmeticOp::MULTIPLICATION, trueValue);
  }

  // default case: trueValue != 0 && falseValue != 0 => value is changed in both branches of If statement
  // -> return condition*trueValue + (1-b)*falseValue.
  // factorIsTrue = ifStatementCondition
  auto factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();
  // factorIsFalse = [1-ifStatementCondition]
  auto factorIsFalse = new ArithmeticExpr(new LiteralInt(1),
                                          ArithmeticOp::SUBTRACTION,
                                          condition->clone(false)->castTo<AbstractExpr>());
  return new ArithmeticExpr(
      new ArithmeticExpr(factorIsTrue,
                         ArithmeticOp::MULTIPLICATION,
                         trueValue->clone(false)->castTo<AbstractExpr>()),
      ArithmeticOp::ADDITION,
      new ArithmeticExpr(factorIsFalse,
                         ArithmeticOp::MULTIPLICATION,
                         falseValue->clone(false)->castTo<AbstractExpr>()));
}

void CompileTimeExpressionSimplifier::cleanUpAfterStatementVisited(AbstractNode *statement,
                                                                   bool enqueueStatementForDeletion) {
  // mark this statement for deletion as we don't need it anymore
  if (enqueueStatementForDeletion) nodesQueuedForDeletion.push_back(statement);

  // free the accumulator of binary expressions
  binaryExpressionAccumulator.reset();
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

void CompileTimeExpressionSimplifier::validateAst(AbstractNode *rootNode) {
  // a helper utility
  auto printWarningMsg = [](AbstractNode *node, std::string message) {
    std::cout << "[WARN] Node (" << node->getUniqueNodeId() << "): " << message;
  };
  // an AST can be considered as CTES-valid if all of the following holds:
  // - all edges are non-reversed
  // - all nodes have exactly one parent
  // - all children u of a node v have parent v
  std::deque<AbstractNode *> nodesToProcessNext{rootNode};
  while (!nodesToProcessNext.empty()) {
    // get next node
    auto curNode = nodesToProcessNext.front();
    nodesToProcessNext.pop_front();
    //
    if (curNode->isReversed) printWarningMsg(curNode, "Node is reversed");
    // visit all of the node's children
    for (auto &c : curNode->getChildren()) {
      if (c->getParentsNonNull().size()!=1) printWarningMsg(c, "Node has unexpected number of parents.");
      if (!c->hasParent(curNode)) printWarningMsg(c, "Node does not have expected parent backreference");
      nodesToProcessNext.push_back(c);
    }
  }
}
