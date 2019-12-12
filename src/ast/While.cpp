#include "../../include/ast/While.h"


While::While(AbstractExpr *condition, AbstractStatement *body) : condition(condition), body(body) {

}


json While::toJson() const {
    json j;
    j["type"] = "While";
    j["condition"] = condition->toJson();
    j["body"] = body->toJson();
    return j;
}