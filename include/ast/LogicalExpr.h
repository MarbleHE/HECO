
#ifndef MASTER_THESIS_CODE_LOGICALEXPR_H
#define MASTER_THESIS_CODE_LOGICALEXPR_H


#include "Operator.h"
#include "AbstractExpr.h"

class LogicalExpr : public AbstractExpr {
    std::unique_ptr<AbstractExpr> left;
    LogicalCompOperator op;
    std::unique_ptr<AbstractExpr> right;
public:

    LogicalExpr(const std::unique_ptr<AbstractExpr> left, LogicalCompOperator op,
                const std::unique_ptr<AbstractExpr> right);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_LOGICALEXPR_H
