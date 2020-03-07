#include <iostream>
#include "AbstractStatement.h"
#include "Block.h"
#include "Call.h"
#include "ArithmeticExpr.h"

json AbstractStatement::toJson() const {
  return json({"type", "AbstractStatement"});
}

AbstractBinaryExpr *AbstractStatement::contains(AbstractBinaryExpr *aexpTemplate, ArithmeticExpr *excludedSubtree) {
  return nullptr;
}

std::string AbstractStatement::getVarTargetIdentifier() {
  return std::string();
}

bool AbstractStatement::isEqual(AbstractStatement *as) {
  throw std::runtime_error("Unimplemented AbstractStatement::isEqual.");
}

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj) {
  return outs << obj.toString(false);
}

void to_json(json &j, const AbstractStatement &absStat) {
  j = absStat.toJson();
}

void to_json(json &j, const AbstractStatement *absStat) {
  j = absStat->toJson();
}


