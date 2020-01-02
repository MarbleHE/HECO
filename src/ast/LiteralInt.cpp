#include <iostream>
#include "../../include/ast/LiteralInt.h"

LiteralInt::LiteralInt(int value) : value(value) {}

json LiteralInt::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = this->value;
  return j;
}

int LiteralInt::getValue() const {
  return value;
}

void LiteralInt::accept(Visitor &v) {
  v.visit(*this);
}

std::string LiteralInt::getNodeName() const {
  return "LiteralInt";
}
