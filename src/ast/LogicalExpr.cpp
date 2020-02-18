#include "LogicalExpr.h"
#include "Variable.h"

json LogicalExpr::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["leftOperand"] = getLeft() ? getLeft()->toJson() : "";
  j["operator"] = getOp() ? getOp()->getOperatorString() : "";
  j["rightOperand"] = getRight() ? getRight()->toJson() : "";
  return j;
}

void LogicalExpr::accept(Visitor &v) {
  v.visit(*this);
}

AbstractExpr *LogicalExpr::getLeft() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(0, true));
}

Operator *LogicalExpr::getOp() const {
  return reinterpret_cast<Operator *>(getChildAtIndex(1, true));
}

AbstractExpr *LogicalExpr::getRight() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(2, true));
}

std::string LogicalExpr::getNodeName() const {
  return "LogicalExpr";
}

LogicalExpr::LogicalExpr() {
  setAttributes(nullptr, nullptr, nullptr);
}

LogicalExpr::LogicalExpr(OpSymb::LogCompOp op) {
  setAttributes(nullptr, new Operator(op), nullptr);
}

std::vector<std::string> LogicalExpr::getVariableIdentifiers() {
  auto leftVec = getLeft()->getVariableIdentifiers();
  auto rightVec = getRight()->getVariableIdentifiers();
  leftVec.reserve(leftVec.size() + rightVec.size());
  leftVec.insert(leftVec.end(), rightVec.begin(), rightVec.end());
  return leftVec;
}

LogicalExpr *LogicalExpr::contains(LogicalExpr *lexpTemplate, AbstractExpr *excludedSubtree) {
  if (excludedSubtree!=nullptr && this==excludedSubtree) {
    return nullptr;
  } else {
    bool emptyOrEqualLeft = (!lexpTemplate->getLeft() || lexpTemplate->getLeft()==this->getLeft());
    bool emptyOrEqualRight = (!lexpTemplate->getRight() || lexpTemplate->getRight()==this->getRight());
    bool emptyOrEqualOp = (lexpTemplate->getOp()->isUndefined() || *this->getOp()==*lexpTemplate->getOp());
    return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
  }
}

int LogicalExpr::countByTemplate(AbstractExpr *abstractExpr) {
  // check if abstractExpr is of type BinaryExpr
  if (auto expr = dynamic_cast<LogicalExpr *>(abstractExpr)) {
    // check if current BinaryExpr fulfills requirements of template abstractExpr
    // also check left and right operands for nested BinaryExps
    return (this->contains(expr, nullptr)!=nullptr ? 1 : 0)
        + getLeft()->countByTemplate(abstractExpr)
        + getRight()->countByTemplate(abstractExpr);
  } else {
    return 0;
  }
}

AbstractNode *LogicalExpr::cloneFlat() {
  auto clonedLexp = new LogicalExpr();
  clonedLexp->setUniqueNodeId(this->getUniqueNodeId());
  return clonedLexp;
}

int LogicalExpr::getMaxNumberChildren() {
  return 3;
}

void LogicalExpr::setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand) {
  // update tree structure
  removeChildren();
  addChildren({leftOperand, operatore, rightOperand}, false);
  AbstractNode::addParentTo(this, {leftOperand, operatore, rightOperand});
}

bool LogicalExpr::supportsCircuitMode() {
  return true;
}

LogicalExpr *LogicalExpr::clone(bool keepOriginalUniqueNodeId) {
  auto clonedLogicalExpr =
      new LogicalExpr(getLeft()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                      getOp()->clone(keepOriginalUniqueNodeId)->castTo<Operator>(),
                      getRight()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  return clonedLogicalExpr;
  if (keepOriginalUniqueNodeId) clonedLogicalExpr->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedLogicalExpr->swapChildrenParents();
  return clonedLogicalExpr;
}

