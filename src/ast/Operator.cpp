#include <variant>
#include <typeindex>
#include <iostream>
#include "Operator.h"
#include "LiteralInt.h"
#include "LiteralString.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"

void Operator::accept(Visitor &v) {
  v.visit(*this);
}

Operator::Operator(OpSymb::LogCompOp op) : operatorString(OpSymb::getTextRepr(op)), operatorSymbol(op) {}

Operator::Operator(OpSymb::BinaryOp op) : operatorString(OpSymb::getTextRepr(op)), operatorSymbol(op) {}

Operator::Operator(OpSymb::UnaryOp op) : operatorString(OpSymb::getTextRepr(op)), operatorSymbol(op) {}

const std::string &Operator::getOperatorString() const {
  return operatorString;
}

std::string Operator::getNodeName() const {
  return "Operator";
}

bool Operator::isUndefined() {
  return operatorString.empty();
}

bool Operator::operator==(const Operator &rhs) const {
  return operatorString == rhs.operatorString;
}

bool Operator::operator!=(const Operator &rhs) const {
  return !(rhs == *this);
}

bool Operator::equals(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> op) const {
  return this->getOperatorString() == OpSymb::getTextRepr(op);
}

Literal* Operator::applyOperator(Literal* rhs) {
  // determine Literal subtype of rhs
  if (auto rhsString = dynamic_cast<LiteralString*>(rhs))
    return applyOperator(rhsString);
  else if (auto rhsInt = dynamic_cast<LiteralInt*>(rhs))
    return applyOperator(rhsInt);
  else if (auto rhsBool = dynamic_cast<LiteralBool*>(rhs))
    return applyOperator(rhsBool);
  else
    throw std::logic_error("Could not recognize type of lhs in applyOperator(Literal* rhs).");
}

Literal* Operator::applyOperator(LiteralInt* rhs) {
  int value = rhs->getValue();
  if (this->equals(OpSymb::negation)) return new LiteralInt(-value);
  else if (this->equals(OpSymb::increment)) return new LiteralInt(++value);
  else if (this->equals(OpSymb::decrement)) return new LiteralInt(--value);
  else
    throw std::logic_error("Could not apply unary operator (" + this->getOperatorString() + ") on (int).");
}

Literal* Operator::applyOperator(LiteralBool* rhs) {
  bool value = rhs->getValue();
  if (this->equals(OpSymb::negation)) return new LiteralBool(!value);
  else
    throw std::logic_error(
        "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeName() + ").");
}

Literal* Operator::applyOperator(LiteralString* rhs) {
  throw std::logic_error(
      "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeName() + ").");
}

Literal* Operator::applyOperator(LiteralFloat* rhs) {
  throw std::logic_error(
      "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeName() + ").");
}

// -----------------
// First call of applyOperator -> both types are unknown
// -----------------
Literal* Operator::applyOperator(Literal* lhs, Literal* rhs) {
  // determine Literal subtype of lhs
  if (auto lhsString = dynamic_cast<LiteralString*>(lhs))
    return applyOperator(lhsString, rhs);
  else if (auto lhsInt = dynamic_cast<LiteralInt*>(lhs))
    return applyOperator(lhsInt, rhs);
  else if (auto lhsBool = dynamic_cast<LiteralBool*>(lhs))
    return applyOperator(lhsBool, rhs);
  else
    throw std::logic_error("Could not recognize type of lhs in applyOperator(Literal *lhs, Literal *rhs).");
}

// -----------------
// Second call of applyOperator -> the first type is known, the second type is unknown
// -----------------

template<typename A>
Literal* Operator::applyOperator(A* lhs, Literal* rhs) {
  // determine Literal subtype of lhs
  if (auto rhsString = dynamic_cast<LiteralString*>(rhs))
    return applyOperator(lhs, rhsString);
  else if (auto rhsInt = dynamic_cast<LiteralInt*>(rhs))
    return applyOperator(lhs, rhsInt);
  else if (auto rhsBool = dynamic_cast<LiteralBool*>(rhs))
    return applyOperator(lhs, rhsBool);
  else if (auto rhsFloat = dynamic_cast<LiteralFloat*>(rhs))
    return applyOperator(lhs, rhsFloat);
  else
    throw std::logic_error("template<typename A> applyOperator(A* lhs, Literal* rhs) failed!");
}

// -----------------
// Third call of applyOperator -> both types are is known
// -----------------

Literal* Operator::applyOperator(LiteralFloat* lhs, LiteralFloat* rhs) {
  float lhsVal = lhs->getValue();
  float rhsVal = rhs->getValue();

  if (this->equals(OpSymb::addition)) return new LiteralFloat(lhsVal + rhsVal);
  else if (this->equals(OpSymb::subtraction)) return new LiteralFloat(lhsVal - rhsVal);
  else if (this->equals(OpSymb::multiplication)) return new LiteralFloat(lhsVal * rhsVal);
  else if (this->equals(OpSymb::division)) return new LiteralFloat(lhsVal / rhsVal);
  else if (this->equals(OpSymb::modulo)) throw std::logic_error("MOD not supported for (float, float)");

  else if (this->equals(OpSymb::logicalAnd)) throw std::logic_error("AND not supported for (float, float)");
  else if (this->equals(OpSymb::logicalOr)) throw std::logic_error("OR not supported for (float, float)");
  else if (this->equals(OpSymb::logicalXor)) throw std::logic_error("XOR not supported for (float, float)");

  else if (this->equals(OpSymb::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(OpSymb::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(OpSymb::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(OpSymb::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(OpSymb::equal)) return new LiteralBool(lhsVal == rhsVal);
  else if (this->equals(OpSymb::unequal)) return new LiteralBool(lhsVal != rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

Literal* Operator::applyOperator(LiteralFloat* lhs, LiteralInt* rhs) {
  auto rhsFloat = new LiteralFloat(static_cast<float>(rhs->getValue()));
  return applyOperator(lhs, rhsFloat);
}

Literal* Operator::applyOperator(LiteralInt* lhs, LiteralFloat* rhs) {
  auto lhsFloat = new LiteralFloat(static_cast<float>(lhs->getValue()));
  return applyOperator(lhsFloat, rhs);
}

Literal* Operator::applyOperator(LiteralFloat* lhs, LiteralBool* rhs) {
  auto rhsFloat = new LiteralFloat(static_cast<float>(rhs->getValue()));
  return applyOperator(lhs, rhsFloat);
}

Literal* Operator::applyOperator(LiteralBool* lhs, LiteralFloat* rhs) {
  auto lhsFloat = new LiteralFloat(static_cast<float>(lhs->getValue()));
  return applyOperator(lhsFloat, rhs);
}

Literal* Operator::applyOperator(LiteralFloat* lhs, LiteralString* rhs) {
  throw std::invalid_argument("Operators on (float, string) not supported!");

}

Literal* Operator::applyOperator(LiteralString* lhs, LiteralFloat* rhs) {
  throw std::invalid_argument("Operators on (string, float) not supported!");
}

Literal* Operator::applyOperator(LiteralString* lhs, LiteralInt* rhs) {
  throw std::invalid_argument("Operators on (string, int) not supported!");
}

Literal* Operator::applyOperator(LiteralInt* lhs, LiteralString* rhs) {
  throw std::invalid_argument("Operators on (int, string) not supported!");
}

Literal* Operator::applyOperator(LiteralString* lhs, LiteralBool* rhs) {
  throw std::invalid_argument("Operators on (string, bool) not supported!");
}

Literal* Operator::applyOperator(LiteralBool* lhs, LiteralString* rhs) {
  throw std::invalid_argument("Operators on (bool, string) not supported!");
}

Literal* Operator::applyOperator(LiteralString* lhs, LiteralString* rhs) {
  if (this->equals(OpSymb::addition)) return new LiteralString(lhs->getValue() + rhs->getValue());
  else
    throw std::logic_error(getOperatorString() + " not supported for (string, string)");
}

Literal* Operator::applyOperator(LiteralBool* lhs, LiteralInt* rhs) {
  bool lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(OpSymb::addition)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(OpSymb::subtraction)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(OpSymb::multiplication)) return new LiteralInt(lhsVal * rhsVal);
  else if (this->equals(OpSymb::division)) return new LiteralInt(lhsVal / rhsVal);
  else if (this->equals(OpSymb::modulo)) return new LiteralInt(lhsVal % rhsVal);

  else if (this->equals(OpSymb::logicalAnd)) return new LiteralInt(lhsVal && rhsVal);
  else if (this->equals(OpSymb::logicalOr)) return new LiteralInt(lhsVal || rhsVal);
    // see https://stackoverflow.com/a/1596681/3017719
  else if (this->equals(OpSymb::logicalXor)) return new LiteralInt(!(lhsVal) != !(rhsVal));

  else if (this->equals(OpSymb::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(OpSymb::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(OpSymb::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(OpSymb::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(OpSymb::equal)) return new LiteralBool(lhsVal == rhsVal);
  else if (this->equals(OpSymb::unequal)) return new LiteralBool(lhsVal != rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

Literal* Operator::applyOperator(LiteralInt* lhs, LiteralBool* rhs) {
  int lhsVal = lhs->getValue();
  bool rhsVal = rhs->getValue();

  if (this->equals(OpSymb::addition)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(OpSymb::subtraction)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(OpSymb::multiplication)) return new LiteralInt(lhsVal * rhsVal);
  else if (this->equals(OpSymb::division)) return new LiteralInt(lhsVal / rhsVal);
  else if (this->equals(OpSymb::modulo)) return new LiteralInt(lhsVal % rhsVal);

  else if (this->equals(OpSymb::logicalAnd)) return new LiteralInt(lhsVal && rhsVal);
  else if (this->equals(OpSymb::logicalOr)) return new LiteralInt(lhsVal || rhsVal);
    // see https://stackoverflow.com/a/1596681/3017719
  else if (this->equals(OpSymb::logicalXor)) return new LiteralInt(!(lhsVal) != !(rhsVal));

  else if (this->equals(OpSymb::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(OpSymb::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(OpSymb::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(OpSymb::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(OpSymb::equal)) return new LiteralBool(lhsVal == rhsVal);
  else if (this->equals(OpSymb::unequal)) return new LiteralBool(lhsVal != rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

Literal* Operator::applyOperator(LiteralBool* lhs, LiteralBool* rhs) {
  int lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(OpSymb::addition)) return new LiteralBool(lhsVal + rhsVal);
  else if (this->equals(OpSymb::subtraction)) return new LiteralBool(lhsVal - rhsVal);
  else if (this->equals(OpSymb::multiplication)) return new LiteralBool(lhsVal * rhsVal);
  else if (this->equals(OpSymb::division)) return new LiteralBool(lhsVal / rhsVal);
  else if (this->equals(OpSymb::modulo)) return new LiteralBool(lhsVal % rhsVal);

  else if (this->equals(OpSymb::logicalAnd)) return new LiteralBool(lhsVal && rhsVal);
  else if (this->equals(OpSymb::logicalOr)) return new LiteralBool(lhsVal || rhsVal);
  else if (this->equals(OpSymb::logicalXor)) return new LiteralBool(lhsVal != rhsVal);

  else if (this->equals(OpSymb::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(OpSymb::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(OpSymb::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(OpSymb::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(OpSymb::equal)) return new LiteralBool(lhsVal == rhsVal);
  else if (this->equals(OpSymb::unequal)) return new LiteralBool(lhsVal != rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralBool* rhs) failed!");
}

Literal* Operator::applyOperator(LiteralInt* lhs, LiteralInt* rhs) {
  int lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(OpSymb::addition)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(OpSymb::subtraction)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(OpSymb::multiplication)) return new LiteralInt(lhsVal * rhsVal);
  else if (this->equals(OpSymb::division)) return new LiteralInt(lhsVal / rhsVal);
  else if (this->equals(OpSymb::modulo)) return new LiteralInt(lhsVal % rhsVal);

  else if (this->equals(OpSymb::logicalAnd)) throw std::logic_error("AND not supported for (int, int)");
  else if (this->equals(OpSymb::logicalOr)) throw std::logic_error("OR not supported for (int, int)");
  else if (this->equals(OpSymb::logicalXor)) throw std::logic_error("XOR not supported for (int, int)");

  else if (this->equals(OpSymb::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(OpSymb::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(OpSymb::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(OpSymb::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(OpSymb::equal)) return new LiteralBool(lhsVal == rhsVal);
  else if (this->equals(OpSymb::unequal)) return new LiteralBool(lhsVal != rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralInt* lhs, LiteralInt* rhs) failed!");
}

std::string Operator::toString() const {
  return this->getOperatorString();
}

Operator::Operator(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> opVar) {
  this->operatorSymbol = opVar;
  this->operatorString = OpSymb::getTextRepr(opVar);
}

bool Operator::supportsCircuitMode() {
  return true;
}

const std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> &Operator::getOperatorSymbol() const {
  return operatorSymbol;
}

Operator::~Operator() = default;

Node* Operator::createClonedNode(bool keepOriginalUniqueNodeId) {
  return new Operator(this->getOperatorSymbol());
}

