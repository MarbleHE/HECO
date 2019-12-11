#include "../../include/ast/While.h"


While::While(const AbstractExpr *condition, AbstractStatement *body) {

}


json While::toJson() const {
    json j;
    j["type"] = "While";
    j["condition"] = condition->toJson();
    j["body"] = body->toJson();
    return j;
}