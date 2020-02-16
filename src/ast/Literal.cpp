#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralString.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"

Literal::~Literal() = default;

std::ostream &operator<<(std::ostream &os, const Literal &literal) {
  literal.print(os);
  return os;
}

bool Literal::operator==(const Literal &rhs) const {
  // Literals cannot be equal if they have a different type
  if (typeid(*this) != typeid(rhs))
    return false;

  // cast both literals to determine their equivalence
  // note: the dereference op is important here to compare the actual values, not the addresses pointed to
  if (auto thisInt = dynamic_cast<const LiteralInt*>(this)) {
    return *dynamic_cast<const LiteralInt*>(&rhs) == *thisInt;

  } else if (auto thisStr = dynamic_cast<const LiteralString*>(this)) {
    return *dynamic_cast<const LiteralString*>(&rhs) == *thisStr;

  } else if (auto thisBool = dynamic_cast<const LiteralBool*>(this)) {
    return *dynamic_cast<const LiteralBool*>(&rhs) == *thisBool;

  } else if (auto thisFloat = dynamic_cast<const LiteralFloat*>(this)) {
    return *dynamic_cast<const LiteralFloat*>(&rhs) == *thisFloat;

  } else {
    throw std::logic_error("Unexpected Literal type encountered!");
  }
}

bool Literal::operator!=(const Literal &rhs) const {
  return !(rhs == *this);
}

