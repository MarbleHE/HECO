#include <variant>
#include <typeindex>
#include <unordered_set>
#include <iostream>
#include "Operator.h"
#include "LiteralInt.h"
#include "LiteralString.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"
#include "Matrix.h"

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

std::string Operator::getNodeType() const {
  return "Operator";
}

bool Operator::isUndefined() {
  return operatorString.empty();
}

bool Operator::isCommutative() {
  // all commutative operators
  static const std::unordered_set<LogCompOp> logicalOps =
      {LogCompOp::LOGICAL_AND, LogCompOp::LOGICAL_OR, LogCompOp::LOGICAL_XOR};
  static const std::unordered_set<ArithmeticOp> arithmeticOps =
      {ArithmeticOp::ADDITION, ArithmeticOp::MULTIPLICATION};

  auto opSymb = getOperatorSymbol();
  if (std::holds_alternative<ArithmeticOp>(opSymb)) {  // arithmetic operators
    return arithmeticOps.count(std::get<ArithmeticOp>(opSymb)) > 0;
  } else if (std::holds_alternative<LogCompOp>(opSymb)) {  // logical operators
    return logicalOps.count(std::get<LogCompOp>(opSymb)) > 0;
  }
  return false;
}

bool Operator::isLeftAssociative() {
  // all left-associative operators
  static const std::unordered_set<LogCompOp> logicalOps =
      {GREATER, GREATER_EQUAL, SMALLER, SMALLER_EQUAL, EQUAL, UNEQUAL};
  static const std::unordered_set<ArithmeticOp> arithmeticOps =
      {ArithmeticOp::DIVISION, ArithmeticOp::SUBTRACTION};

  auto opSymb = getOperatorSymbol();
  if (std::holds_alternative<ArithmeticOp>(opSymb)) {  // arithmetic operators
    return arithmeticOps.count(std::get<ArithmeticOp>(opSymb)) > 0;
  } else if (std::holds_alternative<LogCompOp>(opSymb)) {  // logical operators
    return logicalOps.count(std::get<LogCompOp>(opSymb)) > 0;
  }
  return false;
}

bool Operator::supportsPartialEvaluation() {
  // all logical operators without relational operators (<, <=, >, >=, !=, ==)
  static const std::unordered_set<LogCompOp> logicalOps =
      {LogCompOp::LOGICAL_AND, LogCompOp::LOGICAL_OR, LogCompOp::LOGICAL_XOR};
  // all arithmetic operators
  static const std::unordered_set<ArithmeticOp> arithmeticOps =
      {ArithmeticOp::ADDITION, ArithmeticOp::SUBTRACTION, ArithmeticOp::MULTIPLICATION, ArithmeticOp::DIVISION,
       ArithmeticOp::MODULO};

  auto opSymb = getOperatorSymbol();
  if (std::holds_alternative<ArithmeticOp>(opSymb)) {  // arithmetic operators
    return arithmeticOps.count(std::get<ArithmeticOp>(opSymb)) > 0;
  } else if (std::holds_alternative<LogCompOp>(opSymb)) {  // logical operators
    return logicalOps.count(std::get<LogCompOp>(opSymb)) > 0;
  }
  return false;
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

std::string Operator::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {this->getOperatorString()});
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
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

// ===============================================================
// Methods for handling unary operator evaluation
// ===============================================================

AbstractLiteral *Operator::applyOperator(AbstractLiteral *rhs) {
  // unary operator on matrix
  if (!rhs->getMatrix()->isScalar()) {
    // clone the existing literal as the result will be of the same type
    auto resultType = rhs->clone(false)->castTo<AbstractLiteral>();
    // store the evaluation result in the new literal
    resultType->setMatrix(rhs->getMatrix()->applyUnaryOperatorComponentwise(this));
    return resultType;
  }

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
  if (this->equals(UnaryOp::NEGATION)) return new LiteralInt(-value);
  else
    throw std::logic_error("Could not apply unary operator (" + this->getOperatorString() + ") on (int).");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *rhs) {
  bool value = rhs->getValue();
  if (this->equals(UnaryOp::NEGATION)) return new LiteralBool(!value);
  else
    throw std::logic_error(
        "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeType() + ").");
}

AbstractLiteral *Operator::applyOperator(LiteralString *) {
  throw std::logic_error(
      "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeType() + ").");
}

AbstractLiteral *Operator::applyOperator(LiteralFloat *) {
  throw std::logic_error(
      "Could not apply unary operator (" + this->getOperatorString() + ") on (" + this->getNodeType() + ").");
}

// ===============================================================
// Methods for handling binary operator evaluation
// ===============================================================

// -----------------
// First call of applyOperator -> both Types are unknown
// -----------------
AbstractLiteral *Operator::applyOperator(AbstractLiteral *lhs, AbstractLiteral *rhs) {
  // if at least one of the operands is a matrix (i.e., non-scalar) -> let Matrix class handle it
  if (!lhs->getMatrix()->isScalar() || !rhs->getMatrix()->isScalar()) {
    // check that type of both operands is the same, in that case also the template type T will be the same
    if (typeid(*lhs)!=typeid(*rhs)) {
      throw std::runtime_error(
          "Operations involving matrices currently only supported for same-type matrices/scalars.");
    }
    // clone the existing literal as the result will be of the same type
    auto resultType = lhs->clone(false)->castTo<AbstractLiteral>();
    // store the evaluation result in the new literal
    // NOTE: applyBinaryOperator cannot handle binary expression with operands of different types yet.
    resultType->setMatrix(lhs->getMatrix()->applyBinaryOperator(rhs->getMatrix(), this));
    return resultType;
  }

  // determine Literal subtype of lhs to continue evaluation chain
  if (auto lhsString = dynamic_cast<LiteralString *>(lhs)) return applyOperator(lhsString, rhs);
  else if (auto lhsInt = dynamic_cast<LiteralInt *>(lhs)) return applyOperator(lhsInt, rhs);
  else if (auto lhsBool = dynamic_cast<LiteralBool *>(lhs)) return applyOperator(lhsBool, rhs);
  else if (auto lhsFloat = dynamic_cast<LiteralFloat *>(lhs)) return applyOperator(lhsFloat, rhs);
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
// Third call of applyOperator -> both Types are known
// -----------------

AbstractLiteral *Operator::applyOperator(LiteralFloat *lhs, LiteralFloat *rhs) {
  float lhsVal = lhs->getValue();
  float rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::ADDITION)) return new LiteralFloat(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::SUBTRACTION)) return new LiteralFloat(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::MULTIPLICATION)) return new LiteralFloat(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::DIVISION)) return new LiteralFloat(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::MODULO)) throw std::logic_error("MOD not supported for (float, float)");

  else if (this->equals(LogCompOp::LOGICAL_AND)) throw std::logic_error("AND not supported for (float, float)");
  else if (this->equals(LogCompOp::LOGICAL_OR)) throw std::logic_error("OR not supported for (float, float)");
  else if (this->equals(LogCompOp::LOGICAL_XOR)) throw std::logic_error("XOR not supported for (float, float)");

  else if (this->equals(LogCompOp::SMALLER)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::SMALLER_EQUAL)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::GREATER)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::GREATER_EQUAL)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::EQUAL)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::UNEQUAL)) return new LiteralBool(lhsVal!=rhsVal);

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
  if (this->equals(ArithmeticOp::ADDITION)) return new LiteralString(lhs->getValue() + rhs->getValue());
  else
    throw std::logic_error(getOperatorString() + " not supported for (string, string)");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *lhs, LiteralInt *rhs) {
  bool lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::ADDITION)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::SUBTRACTION)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::MULTIPLICATION)) return new LiteralInt(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::DIVISION)) return new LiteralInt(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::MODULO)) return new LiteralInt(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::LOGICAL_AND)) return new LiteralInt(lhsVal && rhsVal);
  else if (this->equals(LogCompOp::LOGICAL_OR)) return new LiteralInt(lhsVal || rhsVal);
    // see https://stackoverflow.com/a/1596681/3017719
  else if (this->equals(LogCompOp::LOGICAL_XOR)) return new LiteralInt(!(lhsVal)!=!(rhsVal));

  else if (this->equals(LogCompOp::SMALLER)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::SMALLER_EQUAL)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::GREATER)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::GREATER_EQUAL)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::EQUAL)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::UNEQUAL)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralInt *lhs, LiteralBool *rhs) {
  int lhsVal = lhs->getValue();
  bool rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::ADDITION)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::SUBTRACTION)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::MULTIPLICATION)) return new LiteralInt(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::DIVISION)) return new LiteralInt(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::MODULO)) return new LiteralInt(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::LOGICAL_AND)) return new LiteralInt(lhsVal && rhsVal);
  else if (this->equals(LogCompOp::LOGICAL_OR)) return new LiteralInt(lhsVal || rhsVal);
    // see https://stackoverflow.com/a/1596681/3017719
  else if (this->equals(LogCompOp::LOGICAL_XOR)) return new LiteralInt(!(lhsVal)!=!(rhsVal));

  else if (this->equals(LogCompOp::SMALLER)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::SMALLER_EQUAL)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::GREATER)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::GREATER_EQUAL)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::EQUAL)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::UNEQUAL)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralInt* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralBool *lhs, LiteralBool *rhs) {
  int lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::ADDITION)) return new LiteralBool(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::SUBTRACTION)) return new LiteralBool(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::MULTIPLICATION)) return new LiteralBool(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::DIVISION)) return new LiteralBool(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::MODULO)) return new LiteralBool(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::LOGICAL_AND)) return new LiteralBool(lhsVal && rhsVal);
  else if (this->equals(LogCompOp::LOGICAL_OR)) return new LiteralBool(lhsVal || rhsVal);
  else if (this->equals(LogCompOp::LOGICAL_XOR) || this->equals(LogCompOp::UNEQUAL))
    return new LiteralBool(lhsVal!=rhsVal);

  else if (this->equals(LogCompOp::SMALLER)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::SMALLER_EQUAL)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::GREATER)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::GREATER_EQUAL)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::EQUAL)) return new LiteralBool(lhsVal==rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralBool* lhs, LiteralBool* rhs) failed!");
}

AbstractLiteral *Operator::applyOperator(LiteralInt *lhs, LiteralInt *rhs) {
  int lhsVal = lhs->getValue();
  int rhsVal = rhs->getValue();

  if (this->equals(ArithmeticOp::ADDITION)) return new LiteralInt(lhsVal + rhsVal);
  else if (this->equals(ArithmeticOp::SUBTRACTION)) return new LiteralInt(lhsVal - rhsVal);
  else if (this->equals(ArithmeticOp::MULTIPLICATION)) return new LiteralInt(lhsVal*rhsVal);
  else if (this->equals(ArithmeticOp::DIVISION)) return new LiteralInt(lhsVal/rhsVal);
  else if (this->equals(ArithmeticOp::MODULO)) return new LiteralInt(lhsVal%rhsVal);

  else if (this->equals(LogCompOp::LOGICAL_AND)) throw std::logic_error("AND not supported for (int, int)");
  else if (this->equals(LogCompOp::LOGICAL_OR)) throw std::logic_error("OR not supported for (int, int)");
  else if (this->equals(LogCompOp::LOGICAL_XOR)) throw std::logic_error("XOR not supported for (int, int)");

  else if (this->equals(LogCompOp::SMALLER)) return new LiteralBool(lhsVal < rhsVal);
  else if (this->equals(LogCompOp::SMALLER_EQUAL)) return new LiteralBool(lhsVal <= rhsVal);
  else if (this->equals(LogCompOp::GREATER)) return new LiteralBool(lhsVal > rhsVal);
  else if (this->equals(LogCompOp::GREATER_EQUAL)) return new LiteralBool(lhsVal >= rhsVal);

  else if (this->equals(LogCompOp::EQUAL)) return new LiteralBool(lhsVal==rhsVal);
  else if (this->equals(LogCompOp::UNEQUAL)) return new LiteralBool(lhsVal!=rhsVal);

  else
    throw std::logic_error("applyOperator(LiteralInt* lhs, LiteralInt* rhs) failed!");
}

bool Operator::isArithmeticOp() const {
  return getOperatorSymbol().index()==0;
}

bool Operator::isLogCompOp() const {
  return getOperatorSymbol().index()==1;
}

bool Operator::isUnaryOp() const {
  return getOperatorSymbol().index()==2;
}

AbstractLiteral *Operator::applyOperator(std::vector<int> operands) {
  if (isArithmeticOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<ArithmeticOp>(getOperatorSymbol())) {
      case ADDITION:return new LiteralInt(acc<int, int>([](int a, int b) { return a + b; }, operands));
      case SUBTRACTION:return new LiteralInt(acc<int, int>([](int a, int b) { return a - b; }, operands));
      case MULTIPLICATION:return new LiteralInt(acc<int, int>([](int a, int b) { return a*b; }, operands));
      case DIVISION:return new LiteralInt(acc<int, int>([](int a, int b) { return a/b; }, operands));
      case MODULO:return new LiteralInt(acc<int, int>([](int a, int b) { return a%b; }, operands));
      default:throw std::logic_error("Invalid arithmetic operator!");
    }
  } else if (isLogCompOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<LogCompOp>(getOperatorSymbol())) {
      case LOGICAL_AND:
      case LOGICAL_OR:
      case LOGICAL_XOR:throw std::logic_error("");
      case SMALLER:return new LiteralBool(appPairwise<int>([](int a, int b) { return a < b; }, operands));
      case SMALLER_EQUAL:return new LiteralBool(appPairwise<int>([](int a, int b) { return a <= b; }, operands));
      case GREATER:return new LiteralBool(appPairwise<int>([](int a, int b) { return a > b; }, operands));
      case GREATER_EQUAL:return new LiteralBool(appPairwise<int>([](int a, int b) { return a >= b; }, operands));
      case EQUAL:return new LiteralBool(appPairwise<int>([](int a, int b) { return a==b; }, operands));
      case UNEQUAL:return new LiteralBool(appPairwise<int>([](int a, int b) { return a!=b; }, operands));
      default:throw std::logic_error("Invalid logical/comparison operator!");
    }
  } else if (isUnaryOp()) {
    // throw an error if more than one operand is given because we cannot return multiple values yet
    if (operands.size()!=1) throw std::logic_error("Unary operator only supported for single operand.");
    switch (std::get<UnaryOp>(getOperatorSymbol())) {
      case NEGATION:return new LiteralInt(-operands.at(0));
      default:throw std::logic_error("Invalid unary operator!");
    }
  }
  throw std::logic_error("Unknown operator to be applied on std::vector<int> operands encountered.");
}

AbstractLiteral *Operator::applyOperator(std::vector<float> operands) {
  if (isArithmeticOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<ArithmeticOp>(getOperatorSymbol())) {
      case ADDITION:return new LiteralInt(acc<float, float>([](float a, float b) { return a + b; }, operands));
      case SUBTRACTION:return new LiteralInt(acc<float, float>([](float a, float b) { return a - b; }, operands));
      case MULTIPLICATION:return new LiteralInt(acc<float, float>([](float a, float b) { return a*b; }, operands));
      case DIVISION:return new LiteralInt(acc<float, float>([](float a, float b) { return a/b; }, operands));
      case MODULO:
      default:throw std::logic_error("Invalid arithmetic operator!");
    }
  } else if (isLogCompOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<LogCompOp>(getOperatorSymbol())) {
      case LOGICAL_AND:
      case LOGICAL_OR:
      case LOGICAL_XOR:throw std::logic_error("");
      case SMALLER:return new LiteralBool(appPairwise<float>([](float a, float b) { return a < b; }, operands));
      case SMALLER_EQUAL:return new LiteralBool(appPairwise<float>([](float a, float b) { return a <= b; }, operands));
      case GREATER:return new LiteralBool(appPairwise<float>([](float a, float b) { return a > b; }, operands));
      case GREATER_EQUAL:return new LiteralBool(appPairwise<float>([](float a, float b) { return a >= b; }, operands));
      case EQUAL:return new LiteralBool(appPairwise<float>([](float a, float b) { return a==b; }, operands));
      case UNEQUAL:return new LiteralBool(appPairwise<float>([](float a, float b) { return a!=b; }, operands));
      default:throw std::logic_error("Invalid logical/comparison operator!");
    }
  } else if (isUnaryOp()) {
    // throw an error if more than one operand is given because we cannot return multiple values yet
    if (operands.size()!=1) throw std::logic_error("Unary operator only supported for single operand.");
    switch (std::get<UnaryOp>(getOperatorSymbol())) {
      case NEGATION:return new LiteralInt(-operands.at(0));
      default:throw std::logic_error("Invalid unary operator!");
    }
  }
  throw std::logic_error("Unknown operator to be applied on std::vector<float> operands encountered.");
}

AbstractLiteral *Operator::applyOperator(std::vector<bool> operands) {
  if (isArithmeticOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<ArithmeticOp>(getOperatorSymbol())) {
      case ADDITION:return new LiteralInt(acc<bool, bool>([](bool a, bool b) { return a + b; }, operands));
      case SUBTRACTION:return new LiteralInt(acc<bool, bool>([](bool a, bool b) { return a - b; }, operands));
      case MULTIPLICATION:return new LiteralInt(acc<bool, bool>([](bool a, bool b) { return a*b; }, operands));
      case DIVISION:return new LiteralInt(acc<bool, bool>([](bool a, bool b) { return a/b; }, operands));
      case MODULO:return new LiteralInt(acc<bool, bool>([](bool a, bool b) { return a%b; }, operands));
      default:throw std::logic_error("Invalid arithmetic operator!");
    }
  } else if (isLogCompOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<LogCompOp>(getOperatorSymbol())) {
      case LOGICAL_AND:return new LiteralBool(acc<bool, bool>([](bool a, bool b) { return a && b; }, operands));
      case LOGICAL_OR:return new LiteralBool(acc<bool, bool>([](bool a, bool b) { return a || b; }, operands));
      case LOGICAL_XOR:
        // see https://stackoverflow.com/a/1596681/3017719
        return new LiteralBool(acc<bool, bool>([](bool a, bool b) { return !a!=!b; }, operands));
      case SMALLER:return new LiteralBool(appPairwise<bool>([](bool a, bool b) { return a < b; }, operands));
      case SMALLER_EQUAL:return new LiteralBool(appPairwise<bool>([](bool a, bool b) { return a <= b; }, operands));
      case GREATER:return new LiteralBool(appPairwise<bool>([](bool a, bool b) { return a > b; }, operands));
      case GREATER_EQUAL:return new LiteralBool(appPairwise<bool>([](bool a, bool b) { return a >= b; }, operands));
      case EQUAL:return new LiteralBool(appPairwise<bool>([](bool a, bool b) { return a==b; }, operands));
      case UNEQUAL:return new LiteralBool(appPairwise<bool>([](bool a, bool b) { return a!=b; }, operands));
      default:throw std::logic_error("Invalid logical/comparison operator!");
    }
  } else if (isUnaryOp()) {
    // throw an error if more than one operand is given because we cannot return multiple values yet
    if (operands.size()!=1) throw std::logic_error("Unary operator only supported for single operand.");
    switch (std::get<UnaryOp>(getOperatorSymbol())) {
      case NEGATION:return new LiteralInt(-operands.at(0));
      default:throw std::logic_error("Invalid unary operator!");
    }
  }
  throw std::logic_error("Unknown operator to be applied on std::vector<bool> operands encountered.");
}

AbstractLiteral *Operator::applyOperator(std::vector<std::string> operands) {
  if (isArithmeticOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<ArithmeticOp>(getOperatorSymbol())) {
      case ADDITION:
        return new LiteralString(acc<std::string, std::string>(
            [](std::string a, std::string b) { return a + b; }, operands));
      case SUBTRACTION:
      case MULTIPLICATION:
      case DIVISION:
      case MODULO:
      default:throw std::logic_error("Invalid arithmetic operator!");
    }
  } else if (isLogCompOp()) {
    // ensure that there are at least 2 operands
    switch (std::get<LogCompOp>(getOperatorSymbol())) {
      case SMALLER:
        return new LiteralBool(appPairwise<std::string>([](std::string a, std::string b) { return a < b; },
                                                        operands));
      case SMALLER_EQUAL:
        return new LiteralBool(appPairwise<std::string>([](std::string a, std::string b) {
          return a <= b;
        }, operands));
      case GREATER:
        return new LiteralBool(appPairwise<std::string>([](std::string a, std::string b) { return a > b; },
                                                        operands));
      case GREATER_EQUAL:
        return new LiteralBool(appPairwise<std::string>([](std::string a, std::string b) {
          return a >= b;
        }, operands));
      case EQUAL:
        return new LiteralBool(appPairwise<std::string>([](std::string a, std::string b) { return a==b; },
                                                        operands));
      case UNEQUAL:
        return new LiteralBool(appPairwise<std::string>([](std::string a, std::string b) { return a!=b; },
                                                        operands));
      case LOGICAL_AND:
      case LOGICAL_OR:
      case LOGICAL_XOR:
      default:throw std::logic_error("Invalid logical/comparison operator!");
    }
  } else if (isUnaryOp()) {
    // throw an error if more than one operand is given because we cannot return multiple values yet
    if (operands.size()!=1) throw std::logic_error("Unary operator not supported on LiteralStrings.");
  }
  throw std::logic_error("Unknown operator to be applied on std::vector<bool> operands encountered.");
}

AbstractLiteral *Operator::applyOperator(std::vector<AbstractLiteral *> operands) {
  // assumption/requirement: all elements in operands have same type
  if (dynamic_cast<LiteralInt *>(operands.at(0))) {
    return applyOperator(convert<LiteralInt, int>(operands));
  } else if (dynamic_cast<LiteralFloat *>(operands.at(0))) {
    return applyOperator(convert<LiteralFloat, float>(operands));
  } else if (dynamic_cast<LiteralBool *>(operands.at(0))) {
    return applyOperator(convert<LiteralBool, bool>(operands));
  } else if (dynamic_cast<LiteralString *>(operands.at(0))) {
    return applyOperator(convert<LiteralString, std::string>(operands));
  } else {
    throw std::logic_error("Operator::apply failed because AbstractLiterals are of unknown type.");
  }
}
