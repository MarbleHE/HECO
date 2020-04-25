#include "OperatorExpr.h"
#include <utility>
#include "Operator.h"
#include "AbstractMatrix.h"

int OperatorExpr::getMaxNumberChildren() {
  return -1;
}

bool OperatorExpr::supportsCircuitMode() {
  return true;
}

OperatorExpr::OperatorExpr(Operator *op) {
  setAttributes(op, {});
}

OperatorExpr::OperatorExpr(Operator *op, std::vector<AbstractExpr *> operands) {
  setAttributes(op, std::move(operands));
}

OperatorExpr::OperatorExpr(AbstractExpr *lhsOperand, Operator *op, AbstractExpr *rhsOperand) {
  setAttributes(op, {lhsOperand, rhsOperand});
}

std::string OperatorExpr::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

OperatorExpr *OperatorExpr::clone(bool keepOriginalUniqueNodeId) {
  // clone operator (child0)
  auto clonedOperator = getOperator()->clone(keepOriginalUniqueNodeId);
  // clone all operands (child1...childN)
  std::vector<AbstractExpr *> clonedAes;
  std::transform(++children.begin(), children.end(), std::back_inserter(clonedAes),
                 [keepOriginalUniqueNodeId](AbstractNode *node) -> AbstractExpr * {
                   return node->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>();
                 });
  auto clonedOperatorExpr = new OperatorExpr(clonedOperator, clonedAes);
  clonedOperatorExpr->updateClone(keepOriginalUniqueNodeId, this);
  return clonedOperatorExpr;
}

void OperatorExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string OperatorExpr::getNodeType() const {
  return std::string("OperatorExpr");
}

void OperatorExpr::addOperand(AbstractExpr *operand) {
  auto newOperator = getOperator();
  std::vector<AbstractExpr *> newOperands = getOperands();
  newOperands.push_back(operand);
  // use the setAttributes method that evaluates operands while adding them
  setAttributes(newOperator, newOperands);
}

void OperatorExpr::setOperator(Operator *op) {
  // child at index 0 is always the operator
  auto curOperator = getChildAtIndex(0);
  replaceChild(curOperator, op);
  delete curOperator;
}

OperatorExpr::OperatorExpr() = default;

void OperatorExpr::setAttributes(Operator *newOperator, std::vector<AbstractExpr *> newOperands) {
  // remove any existing children (i.e., operator and operands)
  removeChildren();
  // add the operator
  addChild(newOperator);

  // The basic idea of this OperatorExpr aggregation logic can be summarized as follow:
  // Adding operands to this OperatorExpr always automatically applies the operator on the known operands
  // (AbstractLiterals) such that the number of stored operands is as small as possible.
  // - If the operator is commutative, i.e., the operand order is independent then all known operands are collected and
  // the operator is applied on these known operands. Thereafter, the computed result and the unknown operands are
  // added.
  // - If the operator is left-associative, i.e., the operand order is important (e.g., 24 / 2 / 6), we start collecting
  // operands from the left until encountering the first unknown operand. In that case, we stop and evaluate the
  // operands collected so far. Thereafter these operands are replaced by the evaluation result. This is repeated until
  // reaching the expression's end.
  // - A special case of the left-associative operators are those that cannot be partially evaluated. For example,
  // 31 < 88 < a < 91 would result with the left-associative logic into true < a < 91 which is obviously wrong. Because
  // of that we require that relational operators (<, <=, >, >=, !=, ==) only can be evaluated if all operands are
  // known.

  // if more than two operands are given and the operator is either commutative or left-associative, then aggregate the
  // operands before adding them
  if (newOperands.size() >= 2 && (getOperator()->isCommutative() || getOperator()->isLeftAssociative())) {
    // a vector of the operands to be finally added to this OperatorExpr
    std::vector<AbstractExpr *> simplifiedAbstractExprs;
    // a vector containing those operands that can be aggregated (AbstractLiterals)
    std::vector<AbstractLiteral *> tempAggregator;

    if (getOperator()->isCommutative()) {
      // if operator is commutative: collect all known operands from current OperatorExpr, independent of their position
      // within the expression
      for (auto &c : newOperands) {
        auto valueAsAbstractLiteral = dynamic_cast<AbstractLiteral *>(c);
        if (valueAsAbstractLiteral!=nullptr
            && valueAsAbstractLiteral->isEqual(OpSymb::getNeutralElement(getOperator()->getOperatorSymbol()))) {
          // if this literal is the operator's neutral element (e.g., 0 for ADDITION), drop this element by not
          // adding it to the new OperatorExpr's operands
          continue;
        } else if (valueAsAbstractLiteral!=nullptr && !valueAsAbstractLiteral->getMatrix()->containsAbstractExprs()) {
          // if this is an AbstractLiteral not containing AbstractExprs (but primitive values such as int, float),
          // then we can use that AbstractLiteral for applying the operator on it
          tempAggregator.push_back(valueAsAbstractLiteral);
        } else {
          // otherwise we just transfer the operand to the simplifiedAbstractExprs vector as we cannot aggregate it
          simplifiedAbstractExprs.push_back(c);
        }
      }

      // after processing all operands, we either apply the operator (if at least 2 operands are collected), or
      // we just move the operands to the simplifiedAbstractExprs vector
      if (tempAggregator.size() >= 2) {
        simplifiedAbstractExprs.push_back(getOperator()->applyOperator(tempAggregator));
      } else {
        simplifiedAbstractExprs.insert(simplifiedAbstractExprs.end(), tempAggregator.begin(), tempAggregator.end());
      }

    } else if (getOperator()->isLeftAssociative()) {
      // if operator is associative, collect from left-to-right until first unknown value (e.g., Variable) encountered
      for (auto it = newOperands.begin(); it!=newOperands.end(); ++it) {
        // the currently processed operand
        auto curOperand = dynamic_cast<AbstractLiteral *>(*it);
        // if this is the last processed operand, in this case we need to check whether partial evaluation is applicable
        // or otherwise output the operands collected so far
        bool isLastElement = (it==newOperands.end() - 1);
        // indicates whether we stopped collecting because we encountered an unknown operand (e.g., Variable)
        bool operandCouldNotBeAdded = true;
        // a value can be aggregated if it's an AbstractLiteral of a primitive, e.g., Matrix<int>
        if (curOperand!=nullptr && !curOperand->getMatrix()->containsAbstractExprs()) {
          tempAggregator.push_back(curOperand);
          operandCouldNotBeAdded = false;
        }
        // we need to partially evaluate if we encountered an unknown operand (operandCouldNotBeAdded) or if this is the
        // last operand in the expression (isLastElement)
        if (operandCouldNotBeAdded || isLastElement) {
          if ((tempAggregator.size() >= 2 && getOperator()->supportsPartialEvaluation())
              || (!getOperator()->supportsPartialEvaluation() && tempAggregator.size()==newOperands.size())) {
            // if we collected more than two operands and this operator supports partial evaluation (e.g., 12 / 4 / 3),
            // then we can partially evaluate and store the result; if, however, this operator does not support partial
            // evaluation (e.g., 3 < 11 < 21), we need to make sure that we collected all operands prior evaluation
            simplifiedAbstractExprs.push_back(getOperator()->applyOperator(tempAggregator));
          } else {
            // if we cannot do partial evaluation, just add all collected operands
            simplifiedAbstractExprs.insert(simplifiedAbstractExprs.end(), tempAggregator.begin(), tempAggregator.end());
          }
          // if we broke up collecting operands because we could not add the current operand, we now need to add it
          if (operandCouldNotBeAdded) simplifiedAbstractExprs.push_back(*it);
          tempAggregator.clear();
        }
      }
    } // end of: else if (getOperator()->isLeftAssociative())
    // add the aggregated/simplified operands
    std::vector<AbstractNode *> abstractExprsVec(simplifiedAbstractExprs.begin(), simplifiedAbstractExprs.end());
    addChildren(abstractExprsVec, true);
  } else {
    // add the operands without any prior aggregation
    std::vector<AbstractNode *> abstractExprsVec(newOperands.begin(), newOperands.end());
    addChildren(abstractExprsVec, true);
  }
}

bool OperatorExpr::isLogicalExpr() const {
  return getOperator()->isLogCompOp();
}

Operator *OperatorExpr::getOperator() const {
  return reinterpret_cast<Operator *>(getChildAtIndex(0));
}

bool OperatorExpr::isArithmeticExpr() const {
  return getOperator()->isArithmeticOp();
}

bool OperatorExpr::isUnaryExpr() const {
  return getOperator()->isUnaryOp();
}

bool OperatorExpr::isEqual(AbstractExpr *other) {
  if (auto expr = dynamic_cast<OperatorExpr *>(other)) {
    if (this->getChildren().size()!=other->getChildren().size()) return false;
    if (!this->getOperator()->equals(expr->getOperator()->getOperatorSymbol())) return false;
    for (unsigned int i = 0; i < getOperands().size(); ++i) {
      if (!getOperands().at(i)->isEqual(expr->getOperands().at(i))) return false;
    }
    return true;
  }
  return false;
}

std::vector<AbstractExpr *> OperatorExpr::getOperands() const {
  std::vector<AbstractExpr *> operands;
  // ++children.begin() because operands start from index 1
  std::transform(++children.begin(), children.end(), std::back_inserter(operands),
                 [](AbstractNode *node) -> AbstractExpr * {
                   return node->castTo<AbstractExpr>();
                 });
  return operands;
}

AbstractExpr *OperatorExpr::getRight() const {
  if (getOperands().size() > 2) {
    throw std::logic_error("OperatorExpr::getRight() only supported for expressions with two operands!");
  }
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(2));
}

AbstractExpr *OperatorExpr::getLeft() const {
  if (getOperands().size() > 2) {
    throw std::logic_error("OperatorExpr::getLeft() only supported for expressions with two operands!");
  }
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1));
}

void OperatorExpr::replaceChild(AbstractNode *originalChild, AbstractNode *newChildToBeAdded) {
  // use the standard routine to replace the given child
  AbstractNode::replaceChild(originalChild, newChildToBeAdded);
  auto operands = getOperands();
  if (operands.size() > 1) {
    // apply the operand aggregation mechanism again as replacing a child may have generated new aggregation
    // opportunities (e.g., if variable is now a Literal value)
    auto op = getOperator();
    op->removeFromParents();
    for (auto &operand : operands) operand->removeFromParents(true);
    setAttributes(op, operands);
  }
}
