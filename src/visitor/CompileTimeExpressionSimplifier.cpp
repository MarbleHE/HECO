#include <climits>
#include "ast_opt/visitor/ControlFlowGraphVisitor.h"
#include "ast_opt/visitor/SecretTaintingVisitor.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/visitor/CompileTimeExpressionSimplifier.h"
#include "ast_opt/utilities/NodeUtils.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/LogicalExpr.h"
#include "ast_opt/ast/LiteralFloat.h"
#include "ast_opt/ast/VarDecl.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VarAssignm.h"
#include "ast_opt/ast/UnaryExpr.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/ParameterList.h"
#include "ast_opt/ast/CallExternal.h"
#include "ast_opt/ast/While.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/Rotate.h"
#include "ast_opt/ast/Transpose.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/ast/GetMatrixSize.h"
#include "ast_opt/ast/MatrixAssignm.h"

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
  if (hasKnownValue(elem.getOperand()) && hasKnownValue(elem.getRotationFactor())) {
    auto val = getKnownValue(elem.getOperand());
    // we need a AbstractLiteral to be able to perform the rotation
    if (auto valAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(val)) {
      // clone the AbstractLiteral (including its value)
      auto clonedVal = valAsAbstractLiteral->clone(false)->castTo<AbstractLiteral>();
      // perform rotation on the cloned literal
      clonedVal->getMatrix()->rotate(getKnownValue(elem.getRotationFactor())->castTo<LiteralInt>()->getValue(), true);
      // replace this Rotate node by a new node containing the rotated operand
      elem.getOnlyParent()->replaceChild(&elem, clonedVal);
      enqueueNodeForDeletion(&elem);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(Transpose &elem) {
  Visitor::visit(elem);
  // if the Transpose' operand is known at compile-time, we can execute the transpose cmd and replace this node by the
  // transpose result (i.e., transposed operand)
  if (hasKnownValue(elem.getOperand())) {
    auto val = getKnownValue(elem.getOperand());
    // we need a AbstractLiteral to be able to perform the rotation
    if (auto valAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(val)) {
      // clone the AbstractLiteral (including its value)
      auto clonedVal = valAsAbstractLiteral->clone(false)->castTo<AbstractLiteral>();
      // perform transpose on the cloned literal
      clonedVal->getMatrix()->transpose(true);
      // replace this Rotate node by a new node containing the rotated operand
      elem.getOnlyParent()->replaceChild(&elem, clonedVal);
      enqueueNodeForDeletion(&elem);
    }
  }
}

void CompileTimeExpressionSimplifier::visit(MatrixAssignm &elem) {
  // Do not visit the MatrixElementRef because it would replace the node by a copy of the retrieved value but in a
  // MatrixAssignm we need to modify the value at the given position instead. However, our implementation does not
  // allow to retrieve a real (assignable) reference using MatrixElementRef.
//  Visitor::visit(elem);

  // visit the row and column index
  elem.getAssignmTarget()->getRowIndex()->accept(*this);
  elem.getAssignmTarget()->getColumnIndex()->accept(*this);
  elem.getValue()->accept(*this);

  // flag to mark whether to delete this MatrixAssignm node after it has been visited
  bool enqueueNodeForDeletion = false;

  // get operand (matrix) where assignment is targeted to
  auto operandAsVariable = dynamic_cast<Variable *>(elem.getAssignmTarget()->getOperand());
  if (operandAsVariable==nullptr) {
    throw std::logic_error("MatrixAssignm's operand must be a Variable!");
  }

  // check if the given variable was declared-only and not initialized, i.e., the variable refers to a literal that
  // has dimension (0,0)
  auto isNullDimensionLiteral = [&](Variable *var) -> bool {
    auto varEntry = getVariableEntryDeclaredInThisOrOuterScope(operandAsVariable->getIdentifier());
    if (varEntry==variableValues.end()) return false;
    auto literal = dynamic_cast<AbstractLiteral *>(varEntry->second->value);
    if (literal==nullptr) return false;
    return literal->getMatrix()->getDimensions().equals(0, 0);
  };

  if (hasKnownValue(elem.getAssignmTarget()->getRowIndex())
      && hasKnownValue(elem.getAssignmTarget()->getColumnIndex())
      && hasKnownValue(elem.getAssignmTarget()->getOperand())
          // Matrix must either have dimension (0,0) or a value of anything != nullptr, otherwise there was a
          // previous MatrixAssignm that could not be executed, hence it does not make sense to store this assigned value.
      && (isNullDimensionLiteral(operandAsVariable) || getKnownValue(operandAsVariable)!=nullptr)) {
    // if both indices are literals and we know the referred matrix (i.e., is not an input parameter), we can
    // execute the assignment and mark this node for deletion afterwards
    auto rowIdx = getKnownValue(elem.getAssignmTarget()->getRowIndex())->castTo<LiteralInt>()->getValue();
    auto colIdx = getKnownValue(elem.getAssignmTarget()->getColumnIndex())->castTo<LiteralInt>()->getValue();
    setMatrixVariableValue(operandAsVariable->getIdentifier(), rowIdx, colIdx, elem.getValue());

    // clean up removableNodes result from children that indicates whether a child node can safely be deleted
    removableNodes.erase(elem.getAssignmTarget()->getRowIndex());
    removableNodes.erase(elem.getAssignmTarget()->getColumnIndex());

    // this MatrixAssignm has been executed and is not needed anymore
    markNodeAsRemovable(&elem);
    enqueueNodeForDeletion = true;
  } else { // matrix/indices are not known or there was a previous assignment that could not be executed
    auto var = getVariableEntryDeclaredInThisOrOuterScope(operandAsVariable->getIdentifier());
    if (var->second==nullptr) {
      // The Matrix already has the UNKNOWN value (nullptr) assigned, i.e., a MatrixAssignm was visited before that
      // could not be executed as there were unknown indices involved.
      // -> Do nothing as we cannot execute or simplify that MatrixAssignm.
    } else if (var->second->value->castTo<AbstractLiteral>()->getMatrix()->getDimensions().equals(0, 0)) {
      // The variable's value is EMPTY (default-initialized literal without any value), i.e., this is the first
      // MatrixAssignm that could not be executed because there is an unknown index involved.
      // -> Remove the variable's value in variableValues map to prevent execution of any future MatrixAssignms.
      // emit a variable declaration statement if there's none present yet
      if (emittedVariableDeclarations.count(var->first)==0) emitVariableDeclaration(var);
      // and then mark the value as UNKNOWN
      variableValues[var->first] = nullptr;
    } else {
      // The variable's value is set, i.e., a previous MatrixAssignm was executed by CTES. As we now encountered a
      // MatrixAssignm that cannot be executed due to unknown indices, we need to undo the removal of any previous
      // MatrixAssignms that we executed. This is done by emitting the matrix's current value as a VarAssignm.
      // Afterwards, we set the matrix's value to UNKNOWN (nullptr) to prevent execution of any future MatrixAssignms.
      auto varAssignm = emitVariableAssignment(
          getVariableEntryDeclaredInThisOrOuterScope(operandAsVariable->getIdentifier()));

      // and attach the assignment statement immediately before this MatrixAssignm
      elem.getOnlyParent()->addChildren({varAssignm}, true, &elem);

      // and remove the value in variableValues map to avoid saving any further assignments
      variableValues[var->first] = nullptr;
    }
  }
  cleanUpAfterStatementVisited(&elem, enqueueNodeForDeletion);
}

void CompileTimeExpressionSimplifier::visit(MatrixElementRef &elem) {
  Visitor::visit(elem);
  // if this is an expression like "matrix[a][b]" where the operand (matrix) as well as both indices (a,b) are known
  if (hasKnownValue(elem.getOperand()) && hasKnownValue(elem.getRowIndex()) && hasKnownValue(elem.getColumnIndex())) {
    // get the row index
    int rowIndex = getKnownValue(elem.getRowIndex())->castTo<LiteralInt>()->getValue();
    // get the column index
    int columnIndex = getKnownValue(elem.getColumnIndex())->castTo<LiteralInt>()->getValue();
    // get the element at position (row, column)
    auto matrix = dynamic_cast<AbstractLiteral *>(getKnownValue(elem.getOperand()))->getMatrix();
    auto retrievedElement = matrix->getElementAt(rowIndex, columnIndex);
    // replace this MatrixElementRef referred by the parent node by the retrieved element
    elem.getOnlyParent()->replaceChild(&elem, retrievedElement);
  }
}

void CompileTimeExpressionSimplifier::visit(GetMatrixSize &elem) {
  Visitor::visit(elem);

  // if this is an expression like "GetMatrixSize(M, N)" where the matrix M and the dimension N are known
  if (hasKnownValue(elem.getMatrixOperand()) && hasKnownValue(elem.getDimensionParameter())) {
    auto matrix = getKnownValue(elem.getMatrixOperand());
    auto requestedDimension = getKnownValue(elem.getDimensionParameter())->castTo<LiteralInt>();
    auto matrixAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(matrix);
    if (matrixAsAbstractLiteral==nullptr) {
      throw std::logic_error("GetMatrixSize requires an AbstractLiteral subtype as operand. Aborting.");
    }
    auto dimSize = matrixAsAbstractLiteral->getMatrix()->getDimensions()
        .getNthDimensionSize(requestedDimension->getValue());
    elem.getOnlyParent()->replaceChild(&elem, new LiteralInt(dimSize));
  }
}

void CompileTimeExpressionSimplifier::visit(Ast &elem) {
  // clean up the variableValues map from any possible previous run
  variableValues.clear();

  Visitor::visit(elem);
  // Delete all noted queued for deletion after finishing the simplification traversal.
  // It's important that we perform deletion in a FIFO-style because otherwise it can happen that we delete an enclosing
  // statement after trying to delete its child that is still in nodesQueuedForDeletion. However, the child is already
  // non-existent as we performed deletion recursively on the enclosing statement including its whole subtree.
  std::set<AbstractNode *> nodesAlreadyDeleted;

  while (!nodesQueuedForDeletion.empty()) {
    auto nodeToBeDeleted = nodesQueuedForDeletion.front();

    // if nodeToBeDeleted is a VarAssignm, we need to check if this was a emitted VarAssignm
    auto nodeAsVarAssignm = dynamic_cast<VarAssignm *>(nodeToBeDeleted);
    if (nodeAsVarAssignm!=nullptr && emittedVariableAssignms.count(nodeAsVarAssignm) > 0) {
      // update the emittedVariableAssignms by deleting the current node (nodeToBeDeleted)
      auto ref = emittedVariableAssignms.at(nodeAsVarAssignm);
      emittedVariableAssignms.erase(emittedVariableAssignms.find(nodeAsVarAssignm));
      ref->second->removeVarAssignm(nodeAsVarAssignm);
      // if the associated VarDecl does not have any other depending VarAssignms requiring it, we can delete that
      // VarDecl too as we do not need it anymore
      if (ref->second->hasNoVarAssignms()) {
        nodesQueuedForDeletion.push_back(ref->second->getVarDeclStatement());
        emittedVariableAssignms.erase(nodeAsVarAssignm);
      }
    }

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
  // TODO: Introduce a depth threshold (#nodes) to stop inlining if a variable's symbolic value reached a certain depth.
  if (getVariableValueDeclaredInThisOrOuterScope(elem.getIdentifier())!=nullptr && replaceVariablesByValues) {
    // if we know the variable's value (i.e., its value is either any subtype of AbstractLiteral or an AbstractExpr if
    // this is a symbolic value that defines on other variables), we can replace this variable node by its value
    auto variableParent = elem.getOnlyParent();
    auto newValue = getKnownValue(&elem);
    variableParent->replaceChild(&elem, newValue);
  }
}

void CompileTimeExpressionSimplifier::visit(VarDecl &elem) {
  Visitor::visit(elem);
  // determine the variable's value
  AbstractExpr *variableValue;
  auto variableInitializer = elem.getInitializer();
  if (variableInitializer==nullptr) {
    variableValue = AbstractLiteral::createLiteralBasedOnDatatype(elem.getDatatype());
  } else {
    variableValue = variableInitializer;
  }
  // store the variable's value
  addDeclaredVariable(elem.getIdentifier(), elem.getDatatype(), variableValue);
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(variableInitializer);
  // mark this statement as removable as it is deleted
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::visit(VarAssignm &elem) {
  Visitor::visit(elem);
  // store the variable's value
  setVariableValue(elem.getVarTargetIdentifier(), elem.getValue());
  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getValue());
  markNodeAsRemovable(&elem);
  cleanUpAfterStatementVisited(&elem, true);
}

void CompileTimeExpressionSimplifier::visit(ArithmeticExpr &elem) {
  // transform this ArithmeticExpr into an OperatorExpr
  if (!elem.getParentsNonNull().empty()) {
    auto op = elem.getOperator();
    op->removeFromParents();
    std::vector<AbstractExpr *> operands{elem.getLeft(), elem.getRight()};
    elem.getLeft()->removeFromParents();
    elem.getRight()->removeFromParents();
    auto operatorExpr = new OperatorExpr(op, operands);
    elem.getOnlyParent()->replaceChild(&elem, operatorExpr);
    enqueueNodeForDeletion(&elem);
    operatorExpr->accept(*this);
  } else {
    Visitor::visit(elem);
    // clean up removableNodes result from children that indicates whether a child node can safely be deleted
    removableNodes.erase(elem.getLeft());
    removableNodes.erase(elem.getRight());
    removableNodes.erase(elem.getOperator());
  }
}

void CompileTimeExpressionSimplifier::visit(LogicalExpr &elem) {
  // transform this LogicalExpr into an OperatorExpr
  if (!elem.getParentsNonNull().empty()) {
    auto op = elem.getOperator();
    op->removeFromParents();
    std::vector<AbstractExpr *> operands{elem.getLeft(), elem.getRight()};
    elem.getLeft()->removeFromParents();
    elem.getRight()->removeFromParents();
    auto operatorExpr = new OperatorExpr(op, operands);
    elem.getOnlyParent()->replaceChild(&elem, operatorExpr);
    enqueueNodeForDeletion(&elem);
    operatorExpr->accept(*this);
  } else {
    Visitor::visit(elem);
    // clean up removableNodes result from children that indicates whether a child node can safely be deleted
    removableNodes.erase(elem.getLeft());
    removableNodes.erase(elem.getRight());
    removableNodes.erase(elem.getOperator());
  }
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
  } else if (!elem.getParentsNonNull().empty()) {
    // if this UnaryExpr cannot be evaluated, replace the UnaryExpr by an OperatorExpr
    auto op = elem.getOperator();
    op->removeFromParents();
    std::vector<AbstractExpr *> operands{elem.getRight()};
    elem.getRight()->removeFromParents();
    auto operatorExpr = new OperatorExpr(op, operands);
    elem.getOnlyParent()->replaceChild(&elem, operatorExpr);
  }
  enqueueNodeForDeletion(&elem);
}

void CompileTimeExpressionSimplifier::visit(OperatorExpr &elem) {
  // In case that this OperatorExpr has been created recently by transforming an Arithmetic-/LogicalExpr into this
  // OperatorExpr, the operands will be visited again. This is needless but acceptable. In other cases it is
  // important to revisit the operands, for example, if this statement was modified or cloned (e.g., during loop
  // unrolling) and we have new knowledge (e.g., variable values) that must be taken into account while visiting the
  // operands.
  Visitor::visit(elem);

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
  // start by operatorAndOperands.begin() + 1 to skip the first child (operator)
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
      enqueueNodeForDeletion(*it);
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

  // In case that the expression can be simplified, resulting in a single-element expression (e.g., AND false) then it
  // is the caller's responsibility to accordingly replace this OperatorExpr by the only operand.

  if (logicalOperator==LOGICAL_AND) {
    // - knownOperand == false: replace the whole expression by False as <anything> AND False is always False
    // - knownOperand == true: remove True from the expression as <anything> AND True only depends on <anything>
    if (knownOperand->isEqual(new LiteralBool(false))) {
      elem.setAttributes(elem.getOperator(), {knownOperand});
    } else if (knownOperand->isEqual(new LiteralBool(true))) {
      elem.removeChild(knownOperand);
    }
  } else if (logicalOperator==LOGICAL_OR) {
    // - knownOperand == true: replace whole expression by True as <anything> OR True is always True
    // - knownOperand == false: remove False from the expression as <anything> OR False only depends on <anything>
    if (knownOperand->isEqual(new LiteralBool(true))) {
      elem.setAttributes(elem.getOperator(), {knownOperand});
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
    if (!hasKnownValue(fp)) allFunctionParametersAreRemovable = false;
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
  if (hasKnownValue(elem.getParameterList()) && hasKnownValue(elem.getBody())) {
    markNodeAsRemovable(&elem);
  }

  // clean up removableNodes result from children that indicates whether a child node can safely be deleted
  removableNodes.erase(elem.getParameterList());
  removableNodes.erase(elem.getBody());
  cleanUpAfterStatementVisited(&elem, false);
}

void CompileTimeExpressionSimplifier::visit(FunctionParameter &elem) {
  Visitor::visit(elem);

  // a FunctionParameter is a kind of variable declaration but instead of a concrete value we need to use a 'nullptr'
  if (auto valueAsVar = dynamic_cast<Variable *>(elem.getValue())) {
    addDeclaredVariable(valueAsVar->getIdentifier(), elem.getDatatype(), nullptr);
  }

  // This simplifier does not care about the variable's datatype, hence we can mark this node as removable. This mark is
  // only relevant in case that this FunctionParameter is part of a Function that is included into a Call
  // statement because Call statements can be replaced by inlining the Function's computation.
  markNodeAsRemovable(&elem);
}

void CompileTimeExpressionSimplifier::visit(If &elem) {
  // Bypass the base Visitor's logic and directly visit the condition only because we need to know whether it is
  // evaluable at runtime (or not) and its result.
  elem.getCondition()->accept(*this);

  // TODO: Rewriting should only happen if the condition is runtime-known and secret.
  //  If the condition is public and runtime-known, the If-statement should NOT be rewritten because can be handled
  //  more efficient by the runtime system. This check requires information from the ControlFlowGraphVisitor that
  //  does not support variable-scoping yet.

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
        enqueueNodeForDeletion(elem.getElseBranch());
        // we also unlink it from the If statement such that it will not be visited
        elem.removeChild(elem.getElseBranch(), true);
      }
    } else {  // the Else-branch is always executed
      // recursively remove the Then-branch (always exists)
      enqueueNodeForDeletion(elem.getThenBranch());
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
      enqueueNodeForDeletion(&elem);
    }
  }
    // ================
    // Case 2: Condition's evaluation result is UNKNOWN at compile-time (i.e., known at runtime)
    // -> rewrite variables values that are modified in either one or both of the If statement's branches such that the
    //    variable's value depends on the If statement's condition evaluation result.
    // ================
  else { // if we don't know the evaluation result of the If statement's condition -> rewrite the If statement
    // create a copy of the variableValues map and removableNodes map
    auto originalVariableValues = getClonedVariableValuesMap();

    // visit the thenBranch and store its modifications
    elem.getThenBranch()->accept(*this);
    auto variableValuesAfterVisitingThen = getClonedVariableValuesMap();

    // check if there is an Else-branch that we need to visit
    if (elem.getElseBranch()!=nullptr) {
      // restore the original map via copy assignment prior visiting Else-branch
      variableValues = originalVariableValues;
      // visit the Else-branch
      elem.getElseBranch()->accept(*this);
    }

    // rewrite those entries that were modified in either one or both maps
    // note: up to this point (and beyond), the Else-branch's modifications are in variableValues
    for (auto &[variableIdentifier, originalValue] : originalVariableValues) {
      // check if the variable was changed in the Then-branch
      auto thenBranchValue = variableValuesAfterVisitingThen.at(variableIdentifier)->value;
      auto thenBranchModifiedCurrentVariable = (thenBranchValue!=originalValue->value);
      // check if the variable was changed in the Else-branch
      // if there is no Else-branch, elseBranchModifiedCurrentVariable stays False
      bool elseBranchModifiedCurrentVariable = false;
      AbstractExpr *elseBranchValue = nullptr;
      if (elem.getElseBranch()!=nullptr) {
        elseBranchValue = variableValues.at(variableIdentifier)->value;
        elseBranchModifiedCurrentVariable = (elseBranchValue!=originalValue->value);
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
        newValue = generateIfDependentValue(elem.getCondition(), thenBranchValue, originalValue->value);
      } else if (elseBranchModifiedCurrentVariable) {
        newValue = generateIfDependentValue(elem.getCondition(), originalValue->value, elseBranchValue);
      } else {
        // otherwise neither one of the two branches modified the variable's value and we can keep it unchanged
        continue;
      }
      // assign the new If statement-dependent value (e.g., myVarIdentifier = condition*32+[1-condition]*11)
      originalVariableValues.at(variableIdentifier)->setValue(newValue);
    }
    // restore the original map that contains the merged changes from the visited branches
    variableValues = originalVariableValues;

    // enqueue the If statement and its children for deletion
    enqueueNodeForDeletion(&elem);
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

std::set<std::pair<std::string, Scope *>>
CompileTimeExpressionSimplifier::removeVarsWrittenAndReadFromVariableValues(Block &blockStmt) {
  // Erase any variable from variableValues that is written AND read in any of the block's statements from the loop's
  // body such that they are not replaced by any value while visiting the body's statements.
  ControlFlowGraphVisitor cfgv;
  cfgv.setIgnoreScope(true); // as we're not using the AST as entry point, the scope logic won't work properly
  // a set of variables that are written by the body's statements
  std::set<std::pair<std::string, Scope *>> writtenVars;
  // for each statement in the For-loop's body, collect all variables that are both read and written
  for (auto &bodyStatement: blockStmt.getStatements()) {
    bodyStatement->accept(cfgv);
    auto varsWrittenAndRead = cfgv.getLastVariablesReadAndWrite();
    // determine the scope for each of the variables
    for (auto &varIdentifier : varsWrittenAndRead) {
      auto varValuesIterator = getVariableEntryDeclaredInThisOrOuterScope(varIdentifier);
      // check if we found the respective variable in the variableValues, otherwise there is no need to add it to
      // writtenVars and to remove its value in variableValues
      if (varValuesIterator!=variableValues.end()) {
        auto scope = varValuesIterator->first.second;
        writtenVars.insert(std::pair(varIdentifier, scope));
      }
    }
  }
  // remove all variables from the current variablesValues that are written in the given Block
  // instead of erasing them, we just set the value to nullptr to keep information such as the scope and datatype
  for (auto &varIdentifierScopePair : writtenVars) {
    variableValues.at(varIdentifierScopePair)->setValue(nullptr);
  }
  return writtenVars;
}

int CompileTimeExpressionSimplifier::determineNumLoopIterations(For &elem) {
  // make a backup of global data structures to allow reverting state
  auto bakNodesQueuedForDeletion = nodesQueuedForDeletion;
  auto bakVariableValues = getClonedVariableValuesMap();

  // run initializer
  if (elem.getInitializer()==nullptr) return -1;
  elem.getInitializer()->accept(*this);

  // check if values of all variables in the condition are known
  bool allVariableHaveKnownValue = true;
  auto variableIdentifiers = elem.getCondition()->getVariableIdentifiers();
  for (auto &var : variableIdentifiers) {
    auto value = getVariableValueDeclaredInThisOrOuterScope(var);
    if (value==nullptr || value==nullptr) {
      allVariableHaveKnownValue = false;
      break; // no need to continue checking other variables
    }
  }

  std::vector<AbstractNode *> deleteNodes;
  int numSimulatedIterations = 0;
  if (allVariableHaveKnownValue) {
    auto conditionEvaluatesTrue = [&]() -> bool {
      auto result = evaluateNodeRecursive(elem.getCondition(), getTransformedVariableMap());
      auto evalResult = result.empty() ? nullptr : result.back();
      if (evalResult==nullptr)
        throw std::runtime_error("Unexpected: Could not evaluate For-loops condition although "
                                 "all variable have known values. Cannot continue.");
      return evalResult->isEqual(new LiteralBool(true));
    };
    auto executeUpdateStmt = [&]() {
      auto clonedUpdateStmt = elem.getUpdateStatement()->clone(false);
      clonedUpdateStmt->accept(*this);
      deleteNodes.push_back(clonedUpdateStmt);
    };
    while (conditionEvaluatesTrue()) {
      numSimulatedIterations++;
      executeUpdateStmt();
    }
  } else {
    // -1 indicates an error, i.e., loop could not be simulated – must be handled by caller properly
    numSimulatedIterations = -1;
  }

  // restore global data structures using backup
  nodesQueuedForDeletion = bakNodesQueuedForDeletion;
  nodesQueuedForDeletion.insert(nodesQueuedForDeletion.end(), deleteNodes.begin(), deleteNodes.end());
  variableValues = bakVariableValues;

  // return counter value
  return numSimulatedIterations;
}

void CompileTimeExpressionSimplifier::visit(For &elem) {
  auto varValuesBackup = getClonedVariableValuesMap();
  auto newNode = handleForLoopUnrolling(elem);
  // After unrolling the outermost loop is finished, visit the loop's body (again) to enable removal of unneeded
  // statements and store latest variable values - this is only required if full loop unrolling was performed. In
  // that case the new node should not be a Block statement. We use that to distinguish between full and partial
  // unrolling.
  if (newNode!=nullptr && dynamic_cast<Block *>(newNode)==nullptr) {
    // Performing the loop unrolling may involve adding temporary variables in the variableValues map. Although they
    // are only valid in the loop's scope, we do not need them as they are substituted. As we are visiting the unrolled
    // statements afterwards next, we must discard these temporary variables by restoring a previously made copy.
    variableValues = varValuesBackup;
    newNode->accept(*this);
  }
}

AbstractNode *CompileTimeExpressionSimplifier::handleForLoopUnrolling(For &elem) {
  // Check if there is another loop in this loop's body, and if yes, visit that loop first.
  // Note: It is not needed to find the innermost loop as this automatically happens thanks to recursion.
  // TODO: This does not work with deeper-nested For-loops, for example, a For-loop that is nested in the Then-branch
  //  of an If-statement.
  auto variableValuesBeforeVisitingNestedLoop = getClonedVariableValuesMap();
  for (auto &node : elem.getStatementToBeExecuted()->castTo<Block>()->getStatements()) {
    if (auto nestedLoop = dynamic_cast<For *>(node)) handleForLoopUnrolling(*nestedLoop);
  }
  variableValues = variableValuesBeforeVisitingNestedLoop;

  // TODO: If condition depends on (reads) a secret variable throw an exception as this is extremely slow.
  //  Cannot be checked yet as SecretTaintingVisitor does not support variable scopes yet.
  if (false) {
    throw std::runtime_error(
        "For-loops containing secret variables are not supported because they cannot efficiently "
        "be unrolled or optimized in any other way. Aborting.");
  }

  // If s For-loop's condition is compile-time known & short enough, do full "symbolic eval/unroll", i.e., no
  // For-loop is left afterwards; otherwise just partially unroll the loop to enable batching by executing
  // computations of multiple loop iterations simultaneously.
  // UNROLL_ITERATIONS_THRESHOLD determines up to which number of iterations a loop is fully unrolled
  const int UNROLL_ITERATIONS_THRESHOLD = 512;
  int numIterations = determineNumLoopIterations(elem);
  // if loop unrolling replaced the For node by a new one, then newNode points to that node, otherwise it is nullptr
  AbstractNode *newNode;
  if (numIterations!=-1 && numIterations < UNROLL_ITERATIONS_THRESHOLD) {
    // do full loop unrolling
    newNode = doFullLoopUnrolling(elem, numIterations);
  } else {
    // do partial unrolling w.r.t. ciphertext slots and add a cleanup loop for handling remaining iterations
    newNode = doPartialLoopUnrolling(elem);
  } // end of full unroll check: if (numIterations!=-1 && numIterations < UNROLL_ITERATIONS_THRESHOLD)
  cleanUpAfterStatementVisited(reinterpret_cast<AbstractNode *>(&elem), false);
  return newNode;
}

AbstractNode *CompileTimeExpressionSimplifier::doFullLoopUnrolling(For &elem, int numLoopIterations) {
  // create a new Block statemeny: necessary to avoid overriding user-defined variables that expect the initalizer
  // to be within the for-loop's scope only
  auto unrolledStatementsBlock = new Block();

  // create copies to allow reverting any changes made by visiting nodes
  auto variableValuesBackup = getClonedVariableValuesMap();
  auto nodesQueuedForDeletionBackup = nodesQueuedForDeletion;

  // move initializer from For-loop into newly added block
  unrolledStatementsBlock->addChild(elem.getInitializer()->removeFromParents(true));
  // visit the initializer to determine the loop variables
  unrolledStatementsBlock->accept(*this);
  auto loopVariablesMap = getChangedVariables(variableValuesBackup);
  variableValues = variableValuesBackup;

  while (numLoopIterations > 0) {
    // for each statement in the for-loop's body
    std::vector<AbstractStatement *>
        stmtsToVisit = elem.getStatementToBeExecuted()->castTo<Block>()->getStatements();
    for (auto it = stmtsToVisit.begin(); it!=stmtsToVisit.end(); ++it) {
      // if this is a block that was created by unrolling a nested loop, we need to consider its containing statements
      if (auto stmtAsBlock = dynamic_cast<Block *>(*it)) {
        auto blockStmts = stmtAsBlock->getStatements();
        it = stmtsToVisit.insert(it + 1, blockStmts.begin(), blockStmts.end());
      }
      // clone the body statement and append the statement to the unrolledForLoop Body
      unrolledStatementsBlock->addChild((*it)->clone(false));
    }
    // add a copy of the update statement, visiting the body then automatically handles the iteration variable in the
    // cloned loop body statements (i.e., no need to manually adapt them)
    unrolledStatementsBlock->addChild(elem.getUpdateStatement()->clone(false));

    numLoopIterations--;
  }

  // remove all variables that are read and written in the body (see ELSE branch)
  removeVarsWrittenAndReadFromVariableValues(*unrolledStatementsBlock);

  // visit the unrolled statements to perform variable substitution and store the most recent variable values
  auto variableValuesBeforeVisitingUnrolledBody = getClonedVariableValuesMap();
  auto nodesQueuedForDeletionBeforeVisitingUnrolledBody = nodesQueuedForDeletion;
  unrolledStatementsBlock->accept(*this);
  // find the nodes that were newly marked for deletion and detach them from their parents because otherwise they will
  // unnecessarily be considered while unrolling any possibly existing outer loop
  auto curNode = nodesQueuedForDeletion.back();
  nodesQueuedForDeletion.pop_back();
  while (curNode!=nodesQueuedForDeletionBackup.back()) {
    curNode->removeFromParents();
    curNode = nodesQueuedForDeletion.back();
    nodesQueuedForDeletion.pop_back();
  }

  // after visiting the block statements, the statements (VarDecl and VarAssignm) are marked for deletion hence we
  // need to emit new statements for every changed variable except the loop iteration variable
  for (auto &[varIdentiferScope, varValue] : getChangedVariables(variableValuesBeforeVisitingUnrolledBody)) {
    auto varIterator = variableValues.find(varIdentiferScope);

    // if this is a loop variable: skip iteration because due to full unrolling we do not need it anymore as it is
    // already substituted by its value in the statements using it
    if (loopVariablesMap.count(varIdentiferScope) > 0) {
      variableValues.erase(varIterator);
    } else {
      // otherwise emit and add a variable assignment to the unrolled body
      unrolledStatementsBlock->addChild(emitVariableAssignment(varIterator));
    }
  }

  // replace the For-loop's body by the unrolled statements
  auto blockStatements = unrolledStatementsBlock->getStatements();
  std::vector<AbstractNode *> statements(blockStatements.begin(), blockStatements.end());
  elem.getOnlyParent()->replaceChildren(&elem, statements);

  return blockStatements.front();
}

AbstractNode *CompileTimeExpressionSimplifier::doPartialLoopUnrolling(For &elem) {
  // create a new Block statemeny: necessary for cleanup loop and to avoid overriding user-defined variables that expect
  // the initalizer to be within the for-loop's scope only
  auto blockEmbeddingLoops = new Block();

  // move initializer from For-loop into newly added block
  auto forLoopInitializer = elem.getInitializer();
  forLoopInitializer->removeFromParents();
  blockEmbeddingLoops->addChild(forLoopInitializer);

  // replace this For-loop in its parent node by the new block and move the For-loop into the block
  elem.getOnlyParent()->replaceChild(&elem, blockEmbeddingLoops);
  blockEmbeddingLoops->addChild(&elem);

  // create copies to allow reverting changes made by visiting nodes
  // - a copy of known variables with their values
  auto variableValuesBackup = getClonedVariableValuesMap();
  // - a copy of all nodes marked for deletion
  auto nodesQueuedForDeletionCopy = nodesQueuedForDeletion;
  // - a copy of the whole For-loop including initializer, condition, update stmt, body (required for cleanup loop)
  auto cleanupForLoop = elem.clone(false)->castTo<For>();
  auto ignrd =
      removeVarsWrittenAndReadFromVariableValues(*cleanupForLoop->getStatementToBeExecuted()->castTo<Block>());
  // visit the condiiton, body, and update statement to make required replacements (e.g., Arithmetic/LogicalExpr to
  // OperatorExpr)
  cleanupForLoop->getCondition()->accept(*this);
  cleanupForLoop->getStatementToBeExecuted()->accept(*this);
  cleanupForLoop->getUpdateStatement()->accept(*this);
  // undo changes made by visiting the condition, body, and update statement: we need them for the cleanup loop and
  // do not want them to be deleted
  variableValues = variableValuesBackup;
  nodesQueuedForDeletion = nodesQueuedForDeletionCopy;

  // visit the intializer
  forLoopInitializer->accept(*this);

  // update the nodesQueuedForDeletion as the initializer's VarDecl will be emitted later by calling
  // emitVariableAssignments
  nodesQueuedForDeletionCopy = nodesQueuedForDeletion;
  auto variableValuesAfterVisitingInitializer = getClonedVariableValuesMap();
  // determine the loop variables, i.e., variables changed in the initializer statement
  auto loopVariablesMap = getChangedVariables(variableValuesBackup);

  // the new for-loop body containing the unrolled statements
  auto unrolledForLoopBody = new Block();
  // Generate the unrolled loop statements that consists of:
  //   <all statements of the body with symbolic loop variable for iteration 1>
  //   <update statement>
  //   <loop condition>   <- this will be moved out of the body into the For-loop's condition field afterwards
  //   ... repeat NUM_CIPHERTEXT_SLOTS times ...
  const int NUM_CIPHERTEXT_SLOTS = 3;
  std::vector<Return *> tempReturnStmts;
  for (int i = 0; i < NUM_CIPHERTEXT_SLOTS; i++) {
    // for each statement in the for-loop's body
    for (auto &stmt : elem.getStatementToBeExecuted()->castTo<Block>()->getStatements()) {
      // clone the body statement and append the statement to the unrolledForLoop Body
      unrolledForLoopBody->addChild(stmt->clone(false));
    }
    // temporarily add the condition such that the variables are replaced (e.g., i < 6 -> i+1 < 6 -> i+2 < 6 -> ...)
    // we use a Return statement here as it does not write anything into the variableValues map
    auto retStmt = new Return(elem.getCondition()->clone(false)->castTo<AbstractExpr>());
    tempReturnStmts.push_back(retStmt);
    unrolledForLoopBody->addChild(retStmt);

    // add a copy of the update statement, visiting the body then automatically handles the iteration variable in the
    // cloned loop body statements - no need to manually adapt them
    unrolledForLoopBody->addChild(elem.getUpdateStatement()->clone(false));
  }
  // replace the for loop's body by the unrolled statements
  elem.replaceChild(elem.getStatementToBeExecuted(), unrolledForLoopBody);

  // delete update statement from loop since it's now incorporated into the body but keep a copy since we need it
  // for the cleanup loop
  elem.getUpdateStatement()->removeFromParents();

  // Erase any variable from variableValues that is written in the loop's body such that it is not replaced by any
  // known value while visiting the body's statements. This includes the iteration variable as we moved the update
  // statement into the loop's body.
  auto readAndWrittenVars =
      removeVarsWrittenAndReadFromVariableValues(*elem.getStatementToBeExecuted()->castTo<Block>());

  // restore the copy, otherwise the initializer visited after creating this copy would be marked for deletion
  nodesQueuedForDeletion = nodesQueuedForDeletionCopy;

  // visit the for-loop's body to do inlining
  auto variableValuesBeforeVisitingLoopBody = getClonedVariableValuesMap();
  elem.getStatementToBeExecuted()->accept(*this);

  // Move the expressions of the temporarily added Return statements into the For-loop's condition by combining all
  // conditions using a logical-AND. Then remove all temporarily added Return statements.
  std::vector<AbstractExpr *> newConds;
  for (auto rStmt : tempReturnStmts) {
    rStmt->removeFromParents(true);
    auto expr = rStmt->getReturnExpressions().front();
    expr->removeFromParents(true);
    newConds.push_back(expr);
    enqueueNodeForDeletion(rStmt);
  }
  auto originalCondition = elem.getCondition();
  elem.replaceChild(originalCondition, new OperatorExpr(new Operator(LOGICAL_AND), newConds));
  enqueueNodeForDeletion(originalCondition);

  // TODO: Future work:  Make this entire thing flexible with regard to num_slots_in_ctxt, i.e., allow changing how long
  //  unrolled loops are. Idea: generate all loops (see below cleanup loop ideas) starting from ludacriously large
  //  number, later disable/delete the ones that are larger than actually selected cipheretxt size determined from
  //  parameters?
  // ISSUE: Scheme parameters might depend on loop length? This is a general issue (no loop bound => no bounded depth)

  // find all variables that were changed in the for-loop's body - even iteration vars (important for condition!) - and
  // emit them, i.e, create a new variable assignment for each variable
  // TODO: Do not emit any variable assignments in the for-loop's body if a variable's maximum depth is reached as this
  //  leads to wrong results. This must be considered when introducing the cut-off for "deep variables".
  auto bodyChangedVariables = getChangedVariables(variableValuesBeforeVisitingLoopBody);
  std::list<AbstractNode *> emittedVarAssignms;
  for (auto it = bodyChangedVariables.begin(); it!=bodyChangedVariables.end(); ++it) {
    // Create exactly one statement for each variable that is changed in the loop's body. This implicitly is the
    // statement with the "highest possible degree" of inlining for this variable. Make sure that we emit the loop
    // variables AFTER the body statements.
    auto emittedVarAssignm = emitVariableAssignment(it);
    if (emittedVarAssignm!=nullptr) {
      if (loopVariablesMap.count(it->first) > 0) {
        // if this is a loop variable - add the statement at the *end* of the body
        emittedVarAssignms.push_back(emittedVarAssignm);
      } else {
        // if this is not a loop variable - add the statement at the *beginning* of the body
        emittedVarAssignms.push_front(emittedVarAssignm);
      }
    }
    // Remove all the variables that were changed within the body from variableValues as inlining them in any statement
    // after/outside the loop does not make any sense. We need to keep the variable and only remove the value by
    // setting it to nullptr, otherwise we'll lose information.
    (*it).second->value = nullptr;
  }
  // append the emitted loop body statements
  elem.getStatementToBeExecuted()->castTo<Block>()->addChildren(
      std::vector<AbstractNode *>(emittedVarAssignms.begin(), emittedVarAssignms.end()), true);

  // TODO: Future work (maybe): for large enough num_..., e.g. 128 or 256, it might make sense to have a binary series
  //  of cleanup loops, e.g., if 127 iterations are left, go to 64-unrolled loop, 32-unrolled loop, etc.
  //  When to cut off? -> empirical decision?

  // Handle remaining iterations using a "cleanup loop": place statements in a separate loop for remaining iterations
  // and attach the generated cleanup loop to the newly added Block. The cleanup loop consists of:
  // - initializer: reused from the unrolled loop as the loop variable is declared out of the first for-loop thus
  // still accessible by the cleanup loop.
  // - condition, update statement, body statements: remain the same as in the original For loop.
  blockEmbeddingLoops->addChild(cleanupForLoop);

  return blockEmbeddingLoops;
}

void CompileTimeExpressionSimplifier::visit(Return &elem) {
  Visitor::visit(elem);
  // simplify return expression: replace each evaluated expression by its evaluation result
  bool allValuesAreKnown = true;
  for (auto &returnExpr : elem.getReturnExpressions()) {
    if (!hasKnownValue(returnExpr)) allValuesAreKnown = false;
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

AbstractExpr *CompileTimeExpressionSimplifier::getVariableValueDeclaredInThisOrOuterScope(std::string variableName) {
  auto resultIt = getVariableEntryDeclaredInThisOrOuterScope(variableName);
  if (resultIt==variableValues.end() || resultIt->second==nullptr) return nullptr;
  return resultIt->second->value;
}

VariableValuesMapType::iterator
CompileTimeExpressionSimplifier::getVariableEntryDeclaredInThisOrOuterScope(std::string variableName) {
  // variables to store the iterator to the declaration that is closest in terms of scopes and the distance between the
  // current scope and the scope of the declaration (e.g., distance is zero iff. both are in the same scope)
  VariableValuesMapType::iterator closestDeclarationIterator;
  int closestDeclarationDistance = INT_MAX;

  // go through all variables declared yet
  for (auto it = variableValues.begin(); it!=variableValues.end(); ++it) {
    // if the variable's identifier ("name") does not match, continue iterating
    if ((*it).first.first!=variableName) continue;
    // check if this variable declaration is valid in the current scope: start from curScope and go the scope hierarchy
    // upwards until the current scope matches the scope of the declaration -> declaration is valid in current scope
    auto scope = curScope;
    int scopeDistance = 0;
    while (scope!=nullptr) {
      // check if the current scope and the scope of the declaration are the same
      if (scope==(*it).first.second) {
        // check if this found variable declaration has a lower scope distance
        if (scopeDistance < closestDeclarationDistance) {
          closestDeclarationDistance = scopeDistance;
          closestDeclarationIterator = it;
          break;
        }
      }
      // go to the next "higher" scope and increment the scope distance
      scope = scope->getOuterScope();
      scopeDistance++;
    }
  }
  // if the bestIteratorDistance has still its default value (INT_MAX), return the variableValue's end iterator,
  // otherwise return the variableValues entry (iterator) that is closest to the current scope
  return (closestDeclarationDistance==INT_MAX) ? variableValues.end() : closestDeclarationIterator;
}

bool CompileTimeExpressionSimplifier::hasKnownValue(AbstractNode *node) {
  // A value is considered as known if...
  // i.) it is a Literal of a concrete type (e.g., not a LiteralInt matrix containing AbstractExprs)
  auto nodeAsLiteral = dynamic_cast<AbstractLiteral *>(node);
  if (nodeAsLiteral!=nullptr /*&&  !nodeAsLiteral->getMatrix()->containsAbstractExprs() */) return true;

  // ii.) it is a variable with a known value (in variableValues)
  if (auto abstractExprAsVariable = dynamic_cast<Variable *>(node)) {
    // check that the variable has a value
    auto var = getVariableValueDeclaredInThisOrOuterScope(abstractExprAsVariable->getIdentifier());
    return var!=nullptr
        // and its value is not symbolic (i.e., contains no variables for which the value is unknown)
        && var->getVariableIdentifiers().empty();
  }
  // ii.) or the node is removable and its value does not matter (hence considered as known)
  return removableNodes.count(node) > 0;
}

void CompileTimeExpressionSimplifier::markNodeAsRemovable(AbstractNode *node) {
  removableNodes.insert(node);
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

AbstractExpr *CompileTimeExpressionSimplifier::getKnownValue(AbstractNode *node) {
  // if node is a Literal: return the node itself
  if (auto nodeAsLiteral = dynamic_cast<AbstractLiteral *>(node)) {
    return nodeAsLiteral;
  }
  // if node is a variable: search the variable's value in the map of known variable values
  auto nodeAsVariable = dynamic_cast<Variable *>(node);
  if (nodeAsVariable!=nullptr) {
    auto val = getVariableValueDeclaredInThisOrOuterScope(nodeAsVariable->getIdentifier());
    if (val!=nullptr) {
      // return a clone of the variable's value
      auto pNode = val->clone(false);
      return dynamic_cast<AbstractExpr *>(pNode);
    }
  }
  // in any other case: throw an error
  std::stringstream ss;
  ss << "Cannot determine value for node " << node->getUniqueNodeId() << ". ";
  ss << "Use the method hasKnownValue before invoking this method.";
  throw std::invalid_argument(ss.str());
}

std::unordered_map<std::string, AbstractLiteral *> CompileTimeExpressionSimplifier::getTransformedVariableMap() {
  std::unordered_map<std::string, AbstractLiteral *> variableMap;
  for (auto &[k, v] : variableValues) {
    if (auto varAsLiteral = dynamic_cast<AbstractLiteral *>(v->value)) {
      variableMap[k.first] = varAsLiteral;
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
  if (enqueueStatementForDeletion) enqueueNodeForDeletion(statement);
}

void CompileTimeExpressionSimplifier::setVariableValue(const std::string &variableIdentifier,
                                                       AbstractExpr *valueAnyLiteralOrAbstractExpr) {
  AbstractExpr *valueToStore = nullptr;
  if (valueAnyLiteralOrAbstractExpr!=nullptr) {
    // clone the given value and detach it from its parent
    valueToStore = valueAnyLiteralOrAbstractExpr->clone(false)->castTo<AbstractExpr>();
    valueToStore->removeFromParents();
  }

  // find the scope this variable was declared in
  auto iterator = getVariableEntryDeclaredInThisOrOuterScope(variableIdentifier);
  // If no scope could be found, this variable cannot be saved. As For-loop unrolling currently makes use of the fact
  // that unknown variables are not replaced, we should not throw any exception here. Maybe the For-loop's case can
  // be handled by using the replaceVariableByValues(=False) flag instead?
  if (iterator==variableValues.end()) return;

  Scope *varDeclScope = iterator->first.second;

  // save the variable's value in the variableValues map
  auto key = std::pair(variableIdentifier, varDeclScope);
  variableValues.at(key)->value = valueToStore;
}

void CompileTimeExpressionSimplifier::setMatrixVariableValue(const std::string &variableIdentifier, int row, int column,
                                                             AbstractExpr *matrixElementValue) {
  AbstractExpr *valueToStore = nullptr;
  if (matrixElementValue!=nullptr) {
    // clone the given value and detach it from its parent
    valueToStore = matrixElementValue->clone(false)->castTo<AbstractExpr>();
    valueToStore->removeFromParents();
  }

  auto iterator = getVariableEntryDeclaredInThisOrOuterScope(variableIdentifier);
  if (iterator->second->value==nullptr) {
    std::stringstream errorMsg;
    errorMsg << "setMatrixValue failed: ";
    errorMsg << "Could not find entry in variableValues for variable identifier " << iterator->first.first << " ";
    errorMsg << "by starting search from scope " << iterator->first.second->getScopeIdentifier() << ".";
    throw std::runtime_error(errorMsg.str());
  }

  // on contrary to simple scalars, we do not need to replace the variable in the variableValues map, instead we
  // need to retrieve the associatd matrix and set the element at the specified (row, column)
  auto literal = dynamic_cast<AbstractLiteral *>(iterator->second->value);
  if (literal==nullptr) {
    std::stringstream errorMsg;
    errorMsg << "setMatrixValue failed: ";
    errorMsg << "Current value of matrix " << iterator->first.first << " ";
    errorMsg << "in variableValues is nullptr. ";
    errorMsg << "This should never happen and indicates that an earlier visited MatrixAssignm could not be executed.";
    errorMsg << "Because of that any future-visited MatrixAssignms should not be executed too.";
    throw std::runtime_error(errorMsg.str());
  }
  // store the value at the given position - matrix must handle indices and make sure that matrix is large enough
  literal->getMatrix()->setElementAt(row, column, valueToStore);
}

void CompileTimeExpressionSimplifier::addDeclaredVariable(
    const std::string varIdentifier, Datatype *dType, AbstractExpr *value) {
  // create a clone of the value to be added to variableValues, otherwise changing the original would also modify the
  // one stored in variableValues
  AbstractExpr *clonedValue = (value==nullptr) ? nullptr : value->clone(false)->castTo<AbstractExpr>();

  // store the value in the variableValues map for further use (e.g., substitution: replacing variable identifiers by
  // the value of the referenced variable)
  variableValues[std::pair(varIdentifier, curScope)] =
      new VariableValue(dType->clone(false)->castTo<Datatype>(), clonedValue);
}

bool CompileTimeExpressionSimplifier::isQueuedForDeletion(const AbstractNode *node) {
  return std::find(nodesQueuedForDeletion.begin(), nodesQueuedForDeletion.end(), node)
      !=nodesQueuedForDeletion.end();
}

VariableValuesMapType CompileTimeExpressionSimplifier::getChangedVariables(
    VariableValuesMapType variableValuesBeforeVisitingNode) {
  // the result list of changed variables with their respective value
  decltype(variableValuesBeforeVisitingNode) changedVariables;
  // Loop through all variables in the current variableValues and check for each if it changed.
  // It is important that we loop through variableValues instead of variableValuesBeforeVisitingNode because there may
  // be newly declared variables.
  for (auto &[varIdentifierScope, varValue] : variableValues) {
    // a variable is changed if it either was added (i.e., declaration of a new variable) or its value was changed

    // check if it is a newly declared variable or an existing one
    auto newDeclaredVariable = variableValuesBeforeVisitingNode.count(varIdentifierScope)==0;
    auto existingVariable = !newDeclaredVariable;

    // check if exactly one of both is a nullptr -> no need to compare their concrete value
    auto anyOfTwoIsNullptr = [&](std::pair<std::string, Scope *> varIdentifierScope, VariableValue *varValue) -> bool {
      return (variableValuesBeforeVisitingNode.at(varIdentifierScope)->value==nullptr)!=(varValue->value==nullptr);
    };

    // check if their value is unequal: compare the value of both but prior to that make sure that value is not nullptr
    auto valueIsUnequal = [&](std::pair<std::string, Scope *> varIdentifierScope, VariableValue *varValue) -> bool {
      return (variableValuesBeforeVisitingNode.at(varIdentifierScope)->value!=nullptr && varValue->value!=nullptr)
          && !variableValuesBeforeVisitingNode.at(varIdentifierScope)->value->isEqual(varValue->value);
    };

    if (newDeclaredVariable
        || (existingVariable
            && (anyOfTwoIsNullptr(varIdentifierScope, varValue) || valueIsUnequal(varIdentifierScope, varValue)))) {
      changedVariables.emplace(varIdentifierScope, varValue);
    }
  }
  return changedVariables;
}

void CompileTimeExpressionSimplifier::visit(AbstractMatrix &elem) {
  Visitor::visit(elem);
}

void CompileTimeExpressionSimplifier::emitVariableDeclaration(VariableValuesMapType::iterator variableToEmit) {
  auto parent = variableToEmit->first.second->getScopeOpener();
  auto children = parent->getChildren();
  auto newVarDecl = new VarDecl(variableToEmit->first.first, variableToEmit->second->datatype);
  // passing position in children vector is req. to prepend the new VarAssignm (i.e., as new first child of parent)
  parent->addChildren({newVarDecl}, true, parent->getChildren().begin());
  emittedVariableDeclarations.emplace(variableToEmit->first, new EmittedVariableData(newVarDecl));
}

VarAssignm *CompileTimeExpressionSimplifier::emitVariableAssignment(VariableValuesMapType::iterator variableToEmit) {
  // if the variable has no value, there's no need to create a variable assignment
  if (variableToEmit->second->value==nullptr) return nullptr;

  // check if a variable declaration statement was emitted before for this variable
  if (emittedVariableDeclarations.count(variableToEmit->first)==0) {
    // if there exists no declaration statement for this variable yet, add a variable declaration statement (without
    // initialization) at the beginning of the scope by prepending a VarDecl statement to the parent of the last
    // statement in the scope - this should generally be a Block statement
    emitVariableDeclaration(variableToEmit);
  }

  auto newVarAssignm = new VarAssignm(variableToEmit->first.first,
                                      variableToEmit->second->value->clone(false)->castTo<AbstractExpr>());
  // add a reference in the list of the associated VarDecl
  emittedVariableDeclarations.at(variableToEmit->first)->addVarAssignm(newVarAssignm);
  // add a reference to link from this VarAssignm to the associated VarDecl
  emittedVariableAssignms[newVarAssignm] = emittedVariableDeclarations.find(variableToEmit->first);
  return newVarAssignm;
}

VariableValuesMapType CompileTimeExpressionSimplifier::getClonedVariableValuesMap() {
  VariableValuesMapType clonedMap;
  // call the object's copy constructor for each VariableValue
  for (auto &[k, v] : variableValues) clonedMap[k] = new VariableValue(*v);
  return clonedMap;
}

void CompileTimeExpressionSimplifier::enqueueNodeForDeletion(AbstractNode *node) {
//  node->removeFromParents();
  nodesQueuedForDeletion.push_back(node);
}
