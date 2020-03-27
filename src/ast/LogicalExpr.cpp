#include "LogicalExpr.h"
#include "Variable.h"

void LogicalExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string LogicalExpr::getNodeType() const {
  return "LogicalExpr";
}

LogicalExpr::LogicalExpr() {
  setAttributes(nullptr, nullptr, nullptr);
}

LogicalExpr::LogicalExpr(LogCompOp op) {
  setAttributes(nullptr, new Operator(op), nullptr);
}

AbstractNode *LogicalExpr::cloneFlat() {
  auto clonedLexp = new LogicalExpr();
  clonedLexp->setUniqueNodeId(this->getUniqueNodeId());
  return clonedLexp;
}

LogicalExpr *LogicalExpr::clone(bool keepOriginalUniqueNodeId) {
  auto clonedLogicalExpr =
      new LogicalExpr(getLeft()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                      getOperator()->clone(keepOriginalUniqueNodeId)->castTo<Operator>(),
                      getRight()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  clonedLogicalExpr->updateClone(keepOriginalUniqueNodeId, this);
  return clonedLogicalExpr;
}

std::string LogicalExpr::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}
