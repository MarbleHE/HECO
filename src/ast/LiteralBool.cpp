#include "../../include/ast/LiteralBool.h"

LiteralBool::LiteralBool(bool value) : value(value) {}

json LiteralBool::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = this->value;
  return j;
}

void LiteralBool::accept(Visitor &v) {
  v.visit(*this);
}

bool LiteralBool::isValue() const {
  return value;
}

std::string LiteralBool::getTextValue() const {
  return isValue() ? "true" : "false";
}

std::string LiteralBool::getNodeName() const {
  return "LiteralBool";
}

