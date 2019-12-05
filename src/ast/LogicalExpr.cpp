#include "../../include/ast/LogicalExpr.h"

LogicalExpr::LogicalExpr(const AbstractExpr &left, const Operator &op, const AbstractExpr &right) : left(left), op(op),
                                                                                                    right(right) {}
