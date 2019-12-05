#ifndef MASTER_THESIS_CODE_EXPRESSIONSTMT_H
#define MASTER_THESIS_CODE_EXPRESSIONSTMT_H


#include "AbstractStatement.h"
#include "AbstractExpr.h"

class ExpressionStmt : public AbstractStatement {
    AbstractExpr expr;
public:
    ExpressionStmt(const AbstractExpr &expr);
};


#endif //MASTER_THESIS_CODE_EXPRESSIONSTMT_H
