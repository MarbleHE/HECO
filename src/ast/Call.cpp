#include "../../include/ast/Call.h"

Call::Call(AbstractExpr *callee, const std::vector<FunctionParameter> &arguments) : callee(callee),
                                                                                    arguments(arguments) {}
