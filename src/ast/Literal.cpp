#include "../../include/ast/Literal.h"
#include "../../include/ast/LiteralInt.h"
#include "../../include/ast/LiteralString.h"
#include "../../include/ast/LiteralBool.h"

Literal::~Literal() {}
std::ostream &operator<<(std::ostream &os, const Literal &literal) {
  literal.print(os);
  return os;
}
