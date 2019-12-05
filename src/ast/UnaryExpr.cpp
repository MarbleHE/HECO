#include "../../include/ast/UnaryExpr.h"

UnaryExpr::UnaryExpr(const std::string &op, const AbstractExpr &right) : op(op), right(right) {}
