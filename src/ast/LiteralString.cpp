#include <utility>

#include "../../include/ast/LiteralString.h"

LiteralString::LiteralString(std::string value) : value(std::move(value)) {}

json LiteralString::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = this->value;
  return j;
}

void LiteralString::accept(Visitor &v) {
  v.visit(*this);
}

const std::string &LiteralString::getValue() const {
  return value;
}

std::string LiteralString::getNodeName() const {
  return "LiteralString";
}

LiteralString::~LiteralString() = default;

Literal* LiteralString::evaluate(Ast &ast) {
  return this;
}

void LiteralString::print(std::ostream &str) const {
  str << this->getValue();
}

bool LiteralString::operator==(const LiteralString &rhs) const {
  return value == rhs.value;
}

bool LiteralString::operator!=(const LiteralString &rhs) const {
  return !(rhs == *this);
}

void LiteralString::storeParameterValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) {
  paramsMap.emplace(identifier, this);
}
