#ifndef MASTER_THESIS_CODE_UNARYEXPR_H
#define MASTER_THESIS_CODE_UNARYEXPR_H


#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr {
    Operator *op;
    AbstractExpr *right;
public:
    UnaryExpr(OpSymb::UnaryOp op, AbstractExpr *right);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

};


#endif //MASTER_THESIS_CODE_UNARYEXPR_H
