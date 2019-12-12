#ifndef MASTER_THESIS_CODE_UNARYEXPR_H
#define MASTER_THESIS_CODE_UNARYEXPR_H


#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr {
    UnaryOperator op;
    AbstractExpr *right;
public:
    UnaryExpr(UnaryOperator op, AbstractExpr *right);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_UNARYEXPR_H
