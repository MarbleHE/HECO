#include "../../include/ast/BinaryExpr.h"

BinaryExpr::BinaryExpr(AbstractExpr *left, const BinaryOperator &op, AbstractExpr *right) : left(left), op(op),
                                                                                            right(right) {}
