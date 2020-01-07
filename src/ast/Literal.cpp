#include "../../include/ast/Literal.h"
#include "../../include/ast/LiteralInt.h"
#include "../../include/ast/LiteralString.h"
#include "../../include/ast/LiteralBool.h"

Literal::~Literal() = default;

std::ostream &operator<<(std::ostream &os, const Literal &literal) {
  literal.print(os);
  return os;
}

bool Literal::operator==(const Literal &rhs) const {
  // Literals cannot be equal if they have a different type
  if (typeid(*this) != typeid(rhs)) return false;

  // cast to
  if (auto literalInt = dynamic_cast<const LiteralInt*>(this)) {
    return dynamic_cast<const LiteralInt*>(this) == literalInt;
  } else if (auto literalStr = dynamic_cast<const LiteralString*>(this)) {
    return dynamic_cast<const LiteralString*>(this) == literalStr;
  } else if (auto literalBool = dynamic_cast<const LiteralBool*>(this)) {
    return dynamic_cast<const LiteralBool*>(this) == literalBool;
  } else {
    throw std::logic_error("Unexpected Literal type encountered!");
  }
}

bool Literal::operator!=(const Literal &rhs) const {
  return !(rhs == *this);
}
