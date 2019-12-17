#include "../../include/ast/While.h"


While::While(AbstractExpr *condition, AbstractStatement *body) : condition(condition), body(body) {

}

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

AbstractExpr *While::getCondition() const {
    return condition;
}

AbstractStatement *While::getBody() const {
    return body;
}

std::string While::getNodeName() const {
    return "While";
}

While::~While() {
    delete condition;
    delete body;
}
