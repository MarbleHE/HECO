#include "../../include/ast/Return.h"

json Return::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = (getReturnExpr() != nullptr) ? getReturnExpr()->toJson() : "";
  return j;
}

Return::Return(AbstractExpr* value) {
  setAttributes(value);
}

void Return::accept(Visitor &v) {
  v.visit(*this);
}

std::string Return::getNodeName() const {
  return "Return";
}

Return::~Return() {
  for (auto &child : getChildren()) delete child;
}

Literal* Return::evaluate(Ast &ast) {
  return this->getReturnExpr()->evaluate(ast);
}

Return::Return() {
  setAttributes(nullptr);
}

int Return::getMaxNumberChildren() {
  return 1;
}

void Return::setAttributes(AbstractExpr* returnExpr) {
  removeChildren();
  addChildren({returnExpr}, false);
  addParentTo(this, {returnExpr});
}

AbstractExpr* Return::getReturnExpr() const {
  return reinterpret_cast<AbstractExpr*>(getChildren().front());
}

bool Return::supportsCircuitMode() {
  return true;
}

Node* Return::cloneRecursiveDeep(bool keepOriginalUniqueNodeId) {
  // clone Return statement and its child recursively
  auto clonedReturnStmt =
      new Return(this->getReturnExpr()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  // if keepOriginalUniqueNodeId is set: copy the ID of this node
  if (keepOriginalUniqueNodeId) {
    clonedReturnStmt->setUniqueNodeId(this->getUniqueNodeId());
  }
  return clonedReturnStmt;
}
