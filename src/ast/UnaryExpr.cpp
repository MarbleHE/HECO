#include "../../include/ast/UnaryExpr.h"

json UnaryExpr::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["operator"] = getOp() ? getOp()->getOperatorString() : "";
  j["rightOperand"] = getRight() ? getRight()->toJson() : "";
  return j;
}

UnaryExpr::UnaryExpr(OpSymb::UnaryOp op, AbstractExpr* right) {
  setAttributes(op, right);
}

void UnaryExpr::accept(Visitor &v) {
  v.visit(*this);
}

Operator* UnaryExpr::getOp() const {
  return reinterpret_cast<Operator*>(getChildAtIndex(0));
}

AbstractExpr* UnaryExpr::getRight() const {
  return reinterpret_cast<AbstractExpr*>(getChildAtIndex(1));
}

std::string UnaryExpr::getNodeName() const {
  return "UnaryExpr";
}

UnaryExpr::~UnaryExpr() {
  for (auto &child : getChildren()) delete child;
}

Literal* UnaryExpr::evaluate(Ast &ast) {
  return this->getOp()->applyOperator(this->getRight()->evaluate(ast));
}

bool UnaryExpr::supportsCircuitMode() {
  return true;
}

int UnaryExpr::getMaxNumberChildren() {
  return 2;
}

void UnaryExpr::setAttributes(OpSymb::UnaryOp op, AbstractExpr* expr) {
  removeChildren();
  auto nodesToBeAdded = std::vector<Node*>({new Operator(op), expr});
  addChildren(nodesToBeAdded);
  Node::addParentTo(this, nodesToBeAdded);
}
