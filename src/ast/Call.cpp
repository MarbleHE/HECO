#include "../../include/ast/Call.h"

Call::Call(const AbstractExpr &callee, const std::vector<AbstractExpr> &arguments) : callee(callee),
                                                                                     arguments(arguments) {}
