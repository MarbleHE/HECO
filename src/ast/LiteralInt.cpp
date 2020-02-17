#include <iostream>
#include "LiteralInt.h"
#include "RandNumGen.h"
#include "AbstractExpr.h"

bool LiteralInt::supportsDatatype(Datatype &datatype) {
  return datatype.getType() == TYPES::INT;
}

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

std::vector<Literal *> LiteralInt::evaluate(Ast &ast) {
  return std::vector<Literal *>({this});
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

void LiteralInt::addLiteralValue(std::string identifier, std::unordered_map<std::string, Literal *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralInt::setValue(int newValue) {
  this->value = newValue;
}

void LiteralInt::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomInt());
}

std::string LiteralInt::toString() const {
  return std::to_string(this->getValue());
}

bool LiteralInt::supportsCircuitMode() {
  return true;
}

AbstractNode *LiteralInt::createClonedNode(bool) {
  return new LiteralInt(this->getValue());
}


