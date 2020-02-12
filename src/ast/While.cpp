#include "While.h"
#include "LiteralBool.h"

While::While(AbstractExpr* condition, AbstractStatement* body) : condition(condition), body(body) {}

json While::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["condition"] = condition->toJson();
  j["body"] = body->toJson();
  return j;
}

void While::accept(Visitor &v) {
  v.visit(*this);
}

AbstractExpr* While::getCondition() const {
  return condition;
}

AbstractStatement* While::getBody() const {
  return body;
}

std::string While::getNodeName() const {
  return "While";
}

While::~While() {
  delete condition;
  delete body;
}

Literal* While::evaluate(Ast &ast) {
  while (*dynamic_cast<LiteralBool*>(getCondition()->evaluate(ast)) == LiteralBool(true)) {
    getBody()->evaluate(ast);
  }
  return nullptr;
}
