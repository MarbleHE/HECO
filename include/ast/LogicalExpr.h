
#ifndef MASTER_THESIS_CODE_LOGICALEXPR_H
#define MASTER_THESIS_CODE_LOGICALEXPR_H


#include "ExpressionStmt.h"
#include "Operator.h"

class LogicalExpr : public AbstractExpr {
    AbstractExpr left;
    Operator op;
    AbstractExpr right;
public:
    LogicalExpr(const AbstractExpr &left, const Operator &op, const AbstractExpr &right);
};


#endif //MASTER_THESIS_CODE_LOGICALEXPR_H
