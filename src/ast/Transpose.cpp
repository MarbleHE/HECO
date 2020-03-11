#include "Transpose.h"

std::string Transpose::getNodeType() const {
  return std::string("Transpose");
}

void Transpose::accept(Visitor &v) {
  v.visit(*this);
}

Transpose::Transpose(AbstractExpr *operand) {
  removeChildren();
  addChild(operand, true);
}

AbstractExpr *Transpose::getOperand() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0));
}

Transpose *Transpose::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new Transpose(getOperand()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

json Transpose::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["operand"] = getOperand()->toJson();
  return j;
}

std::vector<std::string> Transpose::getVariableIdentifiers() {
  return getOperand()->getVariableIdentifiers();
}

bool Transpose::contains(Variable *var) {
  return getOperand()->contains(var);
}

bool Transpose::isEqual(AbstractExpr *other) {
  return AbstractExpr::isEqual(other);
}

int Transpose::getMaxNumberChildren() {
  return 1;
}

std::string Transpose::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

AbstractNode *Transpose::cloneFlat() {
  return new Transpose();
}

bool Transpose::supportsCircuitMode() {
  return true;
}

Transpose::Transpose() = default;
