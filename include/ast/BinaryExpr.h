#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"

class BinaryExpr : public AbstractExpr {
public:
    AbstractExpr *left;
    BinaryOperator op;
    AbstractExpr *right;

    /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
    /// \param left is the left operand of the expression.
    /// \param op is the operator of the expression.
    /// \param right is the right operand of the expression.
    BinaryExpr(AbstractExpr *left, const BinaryOperator &op, AbstractExpr *right);
};


#endif //MASTER_THESIS_CODE_BINARYEXPR_H
