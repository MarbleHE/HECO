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

Operator::Operator(LogCompOp op) : operatorString(OpSymb::getTextRepr(op)) {
  operatorSymbol = OpSymbolVariant(op);
}

Operator::Operator(ArithmeticOp op) : operatorString(OpSymb::getTextRepr(op)) {
  operatorSymbol = OpSymbolVariant(op);
}

Operator::Operator(UnaryOp op) : operatorString(OpSymb::getTextRepr(op)) {
  operatorSymbol = OpSymbolVariant(op);
}

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
  return operatorString==rhs.operatorString;
}

bool Operator::operator!=(const Operator &rhs) const {
  return !(rhs==*this);
}

bool Operator::equals(OpSymbolVariant op) const {
  return this->getOperatorString()==OpSymb::getTextRepr(op);
}

bool Operator::equals(ArithmeticOp op) const {
  return this->getOperatorString()
      ==OpSymb::getTextRepr(OpSymbolVariant(op));
}

bool Operator::equals(LogCompOp op) const {
  return this->getOperatorString()
      ==OpSymb::getTextRepr(OpSymbolVariant(op));
}

bool Operator::equals(UnaryOp op) const {
  return this->getOperatorString()
      ==OpSymb::getTextRepr(OpSymbolVariant(op));
}

AbstractLiteral *Operator::applyOperator(AbstractLiteral *rhs) {
  // determine Literal subtype of rhs
  if (auto rhsString = dynamic_cast<LiteralString *>(rhs))
    return applyOperator(rhsString);
  else if (auto rhsInt = dynamic_cast<LiteralInt *>(rhs))
    return applyOperator(rhsInt);
  else if (auto rhsBool = dynamic_cast<LiteralBool *>(rhs))
    return applyOperator(rhsBool);
  else
    throw std::logic_error("Could not recognize type of lhs in applyOperator(Literal* rhs).");
}

AbstractLiteral *Operator::applyOperator(LiteralInt *rhs) {
  int value = rhs->getValue();
  if (this->equals(UnaryOp::negation)) return new LiteralInt(-value);
  else
    throw std::logic_error("Could not apply unary operator (" + this->getOperatorString() + ") on (int).");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *rhs) {
  bool value = rhs->getValue();
  if (this->equals(UnaryOp::negation)) return new LiteralBool(!value);
  else
    throw std::logic_error(
        "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeName() + ").");
}

AbstractLiteral *Operator::applyOperator(LiteralString *) {
  throw std::logic_error(
      "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeName() + ").");
}

AbstractLiteral *Operator::applyOperator(LiteralFloat *) {
  throw std::logic_error(
      "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeName() + ").");
}

// -----------------
// First call of applyOperator -> both Types are unknown
// -----------------
AbstractLiteral *Operator::applyOperator(AbstractLiteral *lhs, AbstractLiteral *rhs) {
  // determine Literal subtype of lhs
  if (auto lhsString = dynamic_cast<LiteralString *>(lhs))
    return applyOperator(lhsString, rhs);
  else if (auto lhsInt = dynamic_cast<LiteralInt *>(lhs))
    return applyOperator(lhsInt, rhs);
  else if (auto lhsBool = dynamic_cast<LiteralBool *>(lhs))
    return applyOperator(lhsBool, rhs);
  else if (auto lhsFloat = dynamic_cast<LiteralFloat *>(lhs))
    return applyOperator(lhsFloat, rhs);
  else
    throw std::logic_error("Could not recognize type of lhs in applyOperator(Literal *lhs, Literal *rhs).");
}

// -----------------
// Second call of applyOperator -> the first type is known, the second type is unknown
// -----------------

template<typename A>
AbstractLiteral *Operator::applyOperator(A *lhs, AbstractLiteral *rhs) {
  // determine Literal subtype of lhs
  if (auto rhsString = dynamic_cast<LiteralString *>(rhs))
    return applyOperator(lhs, rhsString);
  else if (auto rhsInt = dynamic_cast<LiteralInt *>(rhs))
    return applyOperator(lhs, rhsInt);
  else if (auto rhsBool = dynamic_cast<LiteralBool *>(rhs))
    return applyOperator(lhs, rhsBool);
  else if (auto rhsFloat = dynamic_cast<LiteralFloat *>(rhs))
    return applyOperator(lhs, rhsFloat);
  else
    throw std::logic_error("template<typename A> applyOperator(A* lhs, Literal* rhs) failed!");
}

// -----------------
// Third call of applyOperator -> both Types are is known
// -----------------

AbstractLiteral *Operator::applyOperator(LiteralFloat *lhs, LiteralFloat *rhs) {
  float lhsVal = lhs->getValue();
  float rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::addition)) return new LiteralFloat(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::subtraction)) return new LiteralFloat(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::multiplication)) return new LiteralFloat(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::division)) return new LiteralFloat(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::modulo)) throw std::logic_error("MOD not supported for (float, float)");

  else if (this->equals(LogCompOp::logicalAnd)) throw std::logic_error("AND not supported for (float, float)");
  else if (this->equals(LogCompOp::logicalOr)) throw std::logic_error("OR not supported for (float, float)");
  else if (this->equals(LogCompOp::logicalXor)) throw std::logic_error("XOR not supported for (float, float)");

  else if (this->equals(LogCompOp::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::equal)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::unequal)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralFloat *lhs, LiteralInt *rhs) {
  auto rhsFloat = new LiteralFloat(static_cast<float>(rhs->getValue()));
  return applyOperator(lhs, rhsFloat);
}

AbstractLiteral *Operator::applyOperator(LiteralInt *lhs, LiteralFloat *rhs) {
  auto lhsFloat = new LiteralFloat(static_cast<float>(lhs->getValue()));
  return applyOperator(lhsFloat, rhs);
}

AbstractLiteral *Operator::applyOperator(LiteralFloat *lhs, LiteralBool *rhs) {
  auto rhsFloat = new LiteralFloat(static_cast<float>(rhs->getValue()));
  return applyOperator(lhs, rhsFloat);
}

AbstractLiteral *Operator::applyOperator(LiteralBool *lhs, LiteralFloat *rhs) {
  auto lhsFloat = new LiteralFloat(static_cast<float>(lhs->getValue()));
  return applyOperator(lhsFloat, rhs);
}

AbstractLiteral *Operator::applyOperator(LiteralFloat *, LiteralString *) {
  throw std::invalid_argument("Operators on (float, string) not supported!");
}

AbstractLiteral *Operator::applyOperator(LiteralString *, LiteralFloat *) {
  throw std::invalid_argument("Operators on (string, float) not supported!");
}

AbstractLiteral *Operator::applyOperator(LiteralString *, LiteralInt *) {
  throw std::invalid_argument("Operators on (string, int) not supported!");
}

AbstractLiteral *Operator::applyOperator(LiteralInt *, LiteralString *) {
  throw std::invalid_argument("Operators on (int, string) not supported!");
}

AbstractLiteral *Operator::applyOperator(LiteralString *, LiteralBool *) {
  throw std::invalid_argument("Operators on (string, bool) not supported!");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *, LiteralString *) {
  throw std::invalid_argument("Operators on (bool, string) not supported!");
}

AbstractLiteral *Operator::applyOperator(LiteralString *lhs, LiteralString *rhs) {
  if (this->equals(ArithmeticOp::addition)) return new LiteralString(lhs->getValue() + rhs->getValue());
  else
    throw std::logic_error(getOperatorString() + " not supported for (string, string)");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *lhs, LiteralInt *rhs) {
  bool lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::addition)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::subtraction)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::multiplication)) return new LiteralInt(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::division)) return new LiteralInt(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::modulo)) return new LiteralInt(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::logicalAnd)) return new LiteralInt(lhsVal && rhsVal);
  else if (this->equals(LogCompOp::logicalOr)) return new LiteralInt(lhsVal || rhsVal);
    // see https://stackoverflow.com/a/1596681/3017719
  else if (this->equals(LogCompOp::logicalXor)) return new LiteralInt(!(lhsVal)!=!(rhsVal));

  else if (this->equals(LogCompOp::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::equal)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::unequal)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralInt *lhs, LiteralBool *rhs) {
  int lhsVal = lhs->getValue();
  bool rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::addition)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::subtraction)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::multiplication)) return new LiteralInt(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::division)) return new LiteralInt(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::modulo)) return new LiteralInt(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::logicalAnd)) return new LiteralInt(lhsVal && rhsVal);
  else if (this->equals(LogCompOp::logicalOr)) return new LiteralInt(lhsVal || rhsVal);
    // see https://stackoverflow.com/a/1596681/3017719
  else if (this->equals(LogCompOp::logicalXor)) return new LiteralInt(!(lhsVal)!=!(rhsVal));

  else if (this->equals(LogCompOp::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::equal)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::unequal)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *lhs, LiteralBool *rhs) {
  int lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::addition)) return new LiteralBool(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::subtraction)) return new LiteralBool(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::multiplication)) return new LiteralBool(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::division)) return new LiteralBool(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::modulo)) return new LiteralBool(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::logicalAnd)) return new LiteralBool(lhsVal && rhsVal);
  else if (this->equals(LogCompOp::logicalOr)) return new LiteralBool(lhsVal || rhsVal);
  else if (this->equals(LogCompOp::logicalXor) || this->equals(LogCompOp::unequal))
    return new LiteralBool(lhsVal!=rhsVal);

  else if (this->equals(LogCompOp::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::equal)) return new LiteralBool(lhsVal==rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralBool* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralInt *lhs, LiteralInt *rhs) {
  int lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::addition)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::subtraction)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::multiplication)) return new LiteralInt(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::division)) return new LiteralInt(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::modulo)) return new LiteralInt(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::logicalAnd)) throw std::logic_error("AND not supported for (int, int)");
  else if (this->equals(LogCompOp::logicalOr)) throw std::logic_error("OR not supported for (int, int)");
  else if (this->equals(LogCompOp::logicalXor)) throw std::logic_error("XOR not supported for (int, int)");

  else if (this->equals(LogCompOp::smaller)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::smallerEqual)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::greater)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::greaterEqual)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::equal)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::unequal)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralInt* lhs, LiteralInt* rhs) failed!");
}

std::string Operator::toString() const {
  return this->getOperatorString();
}

Operator::Operator(OpSymbolVariant opVar) {
  this->operatorSymbol = opVar;
  this->operatorString = OpSymb::getTextRepr(opVar);
}

bool Operator::supportsCircuitMode() {
  return true;
}

const OpSymbolVariant &Operator::getOperatorSymbol() const {
  return operatorSymbol;
}

Operator::~Operator() = default;

Operator *Operator::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new Operator(this->getOperatorSymbol());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

