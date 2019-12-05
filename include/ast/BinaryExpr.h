#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "ExpressionStmt.h"
#include "Operator.h"

class BinaryExpr : public AbstractExpr {
    AbstractExpr* left;
    Operator op;
    AbstractExpr* right;
public:
    BinaryExpr(AbstractExpr *left, const Operator &op, AbstractExpr *right);
};


#endif //MASTER_THESIS_CODE_BINARYEXPR_H
