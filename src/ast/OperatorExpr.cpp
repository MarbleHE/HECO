#include <utility>
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/utilities/Operator.h"

int OperatorExpr::getMaxNumberChildren() {
  return -1;
}

OperatorExpr::OperatorExpr(Operator *op) {
  setAttributes(op, {});
}

OperatorExpr::OperatorExpr(Operator *op, std::vector<AbstractExpression *> operands) {
  setAttributes(op, std::move(operands));
}

OperatorExpr::OperatorExpr(AbstractExpression *lhsOperand, Operator *op, AbstractExpression *rhsOperand) {
  setAttributes(op, {lhsOperand, rhsOperand});
}

std::string OperatorExpr::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {});
}

OperatorExpr *OperatorExpr::clone() const {
  // clone operator (child0)
  auto clonedOperator = getOperator()->clone();
  // clone all operands (child1...childN)
  std::vector<AbstractExpression *> clonedAes;
  std::transform(++children.begin(), children.end(), std::back_inserter(clonedAes),
                 [](AbstractNode *node) -> AbstractExpression * {
                   return node->clone()->castTo<AbstractExpression>();
                 });
  auto clonedOperatorExpr = new OperatorExpr(clonedOperator, clonedAes);
  return clonedOperatorExpr;
}

void OperatorExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string OperatorExpr::getNodeType() const {
  return std::string("OperatorExpr");
}

void OperatorExpr::addOperand(AbstractExpression *operand) {
  auto newOperator = getOperator();
  std::vector<AbstractExpression *> newOperands = getOperands();
  newOperands.push_back(operand);
  // use the setAttributes method that evaluates operands while adding them
  setAttributes(newOperator, newOperands);
}

void OperatorExpr::addOperands(std::vector<AbstractExpression *> operands) {
  for (auto &o : operands) {
    addOperand(o);
  }
}

void OperatorExpr::setOperator(Operator *op) {
  // child at index 0 is always the operator
  auto curOperator = children.at(0);
  replaceChild(curOperator, op);
  delete curOperator;
}

OperatorExpr::OperatorExpr() = default;

void OperatorExpr::setAttributes(Operator *newOperator, std::vector<AbstractExpression *> newOperands) {
  // remove any existing children (i.e., operator and operands)
  removeChildren();
  // add the operator
  op = newOperator;
  newOperator->setParent(this);

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
    std::vector<const AbstractExpression *> simplifiedAbstractExprs;
    // a vector containing those operands that can be aggregated (AbstractLiterals)
    std::vector<const AbstractLiteral *> tempAggregator;

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
        } else if (getOperator()->equals(MULTIPLICATION) &&
            valueAsAbstractLiteral!=nullptr && (valueAsAbstractLiteral->isEqual(new LiteralInt(0))
            || valueAsAbstractLiteral->isEqual(new LiteralFloat(0.0f)))) {
          // drop any other operands as <something> * 0 = 0
          simplifiedAbstractExprs.clear();
          simplifiedAbstractExprs.push_back(c);
          // do not process any other operands
          break;
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
    std::vector<AbstractExpression *> abstractExprsVec;
    // clone the operands to match non-const-ness requirements
    for (auto &e: simplifiedAbstractExprs) {
      abstractExprsVec.emplace_back(e->clone());
    }
    addOperands(abstractExprsVec);
    if (getOperands().size()==0) {
      throw std::logic_error("Operator expression reduced to zero operands.");
    } else if (getOperands().size()==1) {
      //Replace myself with operand in parent.
      getParent()->replaceChild(this, getOperands()[0]);
    }
  } else if (newOperands.size()==2) {
    // add the operands without any prior aggregation
    std::vector<AbstractExpression *> abstractExprsVec(newOperands.begin(), newOperands.end());
    // clone the operands to match non-const-ness requirements
    for (auto &e: newOperands) {
      abstractExprsVec.emplace_back(e->clone());
    }
    addOperands(abstractExprsVec);
  } else if (newOperands.size()==1 || newOperands.size()==0) {
    throw std::logic_error("Operator expression with 1 or 0 operands is not valid.");
  } else {
    // add the operands without any prior aggregation
    std::vector<AbstractExpression *> abstractExprsVec(newOperands.begin(), newOperands.end());
    // clone the operands to match non-const-ness requirements
    for (auto &e: newOperands) {
      abstractExprsVec.emplace_back(e->clone());
    }
    addOperands(abstractExprsVec);
  }
  for (auto &c: operands) {
    c->setParent(this);
  }
}

bool OperatorExpr::isLogicalExpr() const {
  return getOperator()->isLogCompOp();
}

Operator *OperatorExpr::getOperator() const {
  return dynamic_cast<Operator *>(children.at(0));
}

bool OperatorExpr::isArithmeticExpr() const {
  return getOperator()->isArithmeticOp();
}

bool OperatorExpr::isUnaryExpr() const {
  return getOperator()->isUnaryOp();
}

bool OperatorExpr::isEqual(AbstractExpression *other) {
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

std::vector<AbstractExpression *> OperatorExpr::getOperands() const {
  return operands;
}

AbstractExpression *OperatorExpr::getRight() const {
  if (getOperands().size() > 2) {
    throw std::logic_error("OperatorExpr::getRight() only supported for expressions with two operands!");
  }
  return dynamic_cast<AbstractExpression *>(children.at(2));
}

AbstractExpression *OperatorExpr::getLeft() const {
  if (getOperands().size() > 2) {
    throw std::logic_error("OperatorExpr::getLeft() only supported for expressions with two operands!");
  }
  return dynamic_cast<AbstractExpression *>(children.at(1));
}

void OperatorExpr::replaceChild(AbstractNode *originalChild, AbstractNode *newChildToBeAdded) {
  // use the standard routine to replace the given child
  AbstractNode::replaceChild(originalChild, newChildToBeAdded);
  auto operands = getOperands();
  if (operands.size() > 1) {
    // apply the operand aggregation mechanism again as replacing a child may have generated new aggregation
    // opportunities (e.g., if variable is now a Literal value)
    auto op = getOperator();
    op->removeFromParent();
    for (auto &operand : operands) operand->takeFromParent();
    setAttributes(op, operands);
  } else if (operands.size()==1) {
    throw std::logic_error("Operator Expression was reduced to single operand.");
  } else {// size == 0
    throw std::logic_error("Operator Expression was reduced to zero operands.");
  }
}
std::vector<std::string> OperatorExpr::getVariableIdentifiers() {
  std::vector<std::string> result;
  for (auto &expr : getOperands()) {
    auto vec = expr->getVariableIdentifiers();
    if (!vec.empty()) {
      result.insert(result.end(), vec.begin(), vec.end());
    }
  }
  return result;
}

std::vector<Variable *> OperatorExpr::getVariables() {
  std::vector<Variable *> result;
  for (auto &expr : getOperands()) {
    auto vec = expr->getVariables();
    if (!vec.empty()) {
      result.insert(result.end(), vec.begin(), vec.end());
    }
  }
  return result;
}

void OperatorExpr::removeOperand(AbstractExpression *operand) {
  auto it = std::find(children.begin(), children.end(), operand);
  if (it!=children.end()) {
    (*it)->takeFromParent();
    // if the node supports an infinite number of children (getMaxNumberChildren() == -1), we can delete the node from
    // the children list, otherwise we just overwrite the slot with a nullptr
    if (this->getMaxNumberChildren()!=-1) {
      *it = nullptr;
    } else {
      children.erase(it);
    }
  }
}

std::vector<AbstractNode *> OperatorExpr::getChildren() {
  //TODO: RETURN SOMETHING USEFUL
  return {};
}
std::vector<const AbstractNode *> OperatorExpr::getChildren() const {
  //TODO: RETURN SOMETHING USEFUL
  return {};
}
void OperatorExpr::removeChildren() {
  //TODO: Actually remove
}


