#ifndef MASTER_THESIS_CODE_LOGICALEXPR_H
#define MASTER_THESIS_CODE_LOGICALEXPR_H


#include "Operator.h"
#include "AbstractExpr.h"

class LogicalExpr : public AbstractExpr {
    AbstractExpr *left;
    LogicalCompOperator op;
    AbstractExpr *right;
public:

    LogicalExpr(AbstractExpr *left, LogicalCompOperator op, AbstractExpr *right);


    json toJson() const;
};


#endif //MASTER_THESIS_CODE_LOGICALEXPR_H
