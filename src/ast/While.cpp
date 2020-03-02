#include <exception>
#include "While.h"
#include "LiteralBool.h"

While::While(AbstractExpr *condition, AbstractStatement *body) : condition(condition), body(body) {}

json While::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["condition"] = condition->toJson();
  j["body"] = body->toJson();
  return j;
}

void While::accept(Visitor &v) {
  v.visit(*this);
}

AbstractExpr *While::getCondition() const {
  return condition;
}

AbstractStatement *While::getBody() const {
  return body;
}

std::string While::getNodeType() const {
  return "While";
}

While::~While() {
  delete condition;
  delete body;
}
While *While::clone(bool keepOriginalUniqueNodeId) {
  //TODO(vianda): Implement clone for While
  throw std::runtime_error("Not implemented");
}
