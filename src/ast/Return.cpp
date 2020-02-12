#include "Return.h"

json Return::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = this->value->toJson();
  return j;
}

Return::Return(AbstractExpr* value) : value(value) {
  this->addChild(value);
  value->addParent(this);
}

void Return::accept(Visitor &v) {
  v.visit(*this);
}

AbstractExpr* Return::getValue() const {
  return value;
}

std::string Return::getNodeName() const {
  return "Return";
}

Return::~Return() {
  delete value;
}

Literal* Return::evaluate(Ast &ast) {
  return this->getValue()->evaluate(ast);
}

Return::Return() {}
