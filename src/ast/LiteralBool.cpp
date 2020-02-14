#include "LiteralBool.h"
#include "RandNumGen.h"

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

bool LiteralBool::getValue() const {
  return value;
}

std::string LiteralBool::getTextValue() const {
  return getValue() ? "true" : "false";
}

std::string LiteralBool::getNodeName() const {
  return "LiteralBool";
}

LiteralBool::~LiteralBool() = default;

Literal* LiteralBool::evaluate(Ast &ast) {
  return this;
}

void LiteralBool::print(std::ostream &str) const {
  str << this->getTextValue();
}

bool LiteralBool::operator==(const LiteralBool &rhs) const {
  return value == rhs.value;
}

bool LiteralBool::operator!=(const LiteralBool &rhs) const {
  return !(rhs == *this);
}

void LiteralBool::addLiteralValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralBool::setValue(bool newValue) {
  this->value = newValue;
}

void LiteralBool::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomBool());
}

std::string LiteralBool::toString() const {
  return this->getTextValue();
}

bool LiteralBool::supportsCircuitMode() {
  return true;
}

bool LiteralBool::supportsDatatype(Datatype &datatype) {
  return datatype.getType() == TYPES::BOOL;
}

Node* LiteralBool::createClonedNode(bool) {
  return new LiteralBool(this->getValue());
}
