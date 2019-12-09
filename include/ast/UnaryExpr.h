
#ifndef MASTER_THESIS_CODE_UNARYEXPR_H
#define MASTER_THESIS_CODE_UNARYEXPR_H


#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr {
    UnaryOperator op;
    AbstractExpr right;
public:
    UnaryExpr(const UnaryOperator &op, const AbstractExpr &right);
};


#endif //MASTER_THESIS_CODE_UNARYEXPR_H
