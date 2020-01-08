#include <iostream>
#include "../../include/ast/LiteralInt.h"
#include "LiteralBool.h"
#include <sstream>
#include "LiteralString.h"

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

LiteralInt::~LiteralInt() = default;

Literal* LiteralInt::evaluate(Ast &ast) {
  return this;
}
LiteralInt LiteralInt::operator+(LiteralInt const &lint) {
  return LiteralInt(this->getValue() + lint.getValue());
}
std::ostream &operator<<(std::ostream &os, const LiteralInt &an_int) {
  os << an_int.getValue();
  return os;
}

void LiteralInt::print(std::ostream &str) const {
  str << this->value;
}
bool LiteralInt::operator==(const LiteralInt &rhs) const {
  return value == rhs.value;
}
bool LiteralInt::operator!=(const LiteralInt &rhs) const {
  return !(rhs == *this);
}

void LiteralInt::storeParameterValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) {
  paramsMap.emplace(identifier, this);
}
