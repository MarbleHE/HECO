#include "../../include/ast/UnaryExpr.h"

UnaryExpr::UnaryExpr(const UnaryOperator &op, const AbstractExpr &right) : op(op), right(right) {}
