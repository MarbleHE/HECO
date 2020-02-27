#include <iostream>
#include "Variable.h"
#include "AbstractExpr.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"
#include "LiteralFloat.h"

json AbstractExpr::toJson() const {
  return json({"type", "AbstractExpr"});
}

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj) {
  return outs << obj.toString();
}

LiteralInt *AbstractExpr::createParam(int i) {
  return new LiteralInt(i);
}

LiteralBool *AbstractExpr::createParam(bool b) {
  return new LiteralBool(b);
}

LiteralString *AbstractExpr::createParam(const char *str) {
  return new LiteralString(str);
}

AbstractExpr *AbstractExpr::createParam(AbstractExpr *abstractExpr) {
  return abstractExpr;
}

AbstractBinaryExpr *AbstractExpr::contains(AbstractBinaryExpr *aexpTemplate, AbstractExpr *excludedSubtree) {
  return nullptr;
}

bool AbstractExpr::contains(Variable *var) {
  return false;
}

bool AbstractExpr::isEqual(AbstractExpr *other) {
  throw std::logic_error("isEqual not implemented for given AbstractExpr subclass.");
}

LiteralFloat *AbstractExpr::createParam(float f) {
  return new LiteralFloat(f);
}

int AbstractExpr::countByTemplate(AbstractExpr *abstractExpr) {
  return 0;
}

std::vector<std::string> AbstractExpr::getVariableIdentifiers() {
  return std::vector<std::string>();
}

AbstractNode *AbstractExpr::createParam(AbstractNode *node) {
  throw std::runtime_error(
      "Method AbstractNode::createParam does not support AbstractNode objects. Did you forget to cast the AbstractNode object?");
}

void to_json(json &j, const AbstractExpr &absExpr) {
  j = absExpr.toJson();
}


