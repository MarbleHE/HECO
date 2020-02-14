#include <iostream>
#include "AbstractStatement.h"
#include "Block.h"
#include "Call.h"
#include "BinaryExpr.h"

std::string AbstractStatement::toString() const {
  return this->toJson().dump();
}

json AbstractStatement::toJson() const {
  return json({"type", "AbstractStatement"});
}

void AbstractStatement::accept(Visitor &v) {
  std::cout << "This shouldn't be executed!" << std::endl;
}

BinaryExpr* AbstractStatement::contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree) {
  return nullptr;
}

std::string AbstractStatement::getVarTargetIdentifier() {
  return std::string();
}

bool AbstractStatement::isEqual(AbstractStatement* as) {
  return false;
}

Literal* AbstractStatement::evaluate(Ast &ast) {
  return nullptr;
}

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj) {
  return outs << obj.toString();
}

void to_json(json &j, const AbstractStatement &absStat) {
  j = absStat.toJson();
}

void to_json(json &j, const AbstractStatement* absStat) {
  j = absStat->toJson();
}


