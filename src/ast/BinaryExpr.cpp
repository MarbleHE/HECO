#include "../../include/ast/BinaryExpr.h"

BinaryExpr::BinaryExpr(AbstractExpr *left, const Operator &op, AbstractExpr *right) : left(left), op(op),
                                                                                      right(right) {}
