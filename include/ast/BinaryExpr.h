#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "ExpressionStmt.h"
#include "Operator.h"

class BinaryExpr : public AbstractExpr {
    AbstractExpr *left;
    Operator op;
    AbstractExpr *right;
public:
    /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
    /// \param left is the left operand of the expression.
    /// \param op is the operator of the expression.
    /// \param right is the right operand of the expression.
    BinaryExpr(AbstractExpr *left, const Operator &op, AbstractExpr *right);
};


#endif //MASTER_THESIS_CODE_BINARYEXPR_H
