#include "../../include/ast/While.h"




While::While(const AbstractExpr &condition, AbstractStatement *body) : condition(condition), body(body) {}
