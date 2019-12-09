#include "../../include/ast/LogicalExpr.h"

LogicalExpr::LogicalExpr(AbstractExpr *left, LogicalCompOperator op, AbstractExpr *right) : left(left), op(op),
                                                                                            right(right) {}
