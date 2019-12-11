#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"

class BinaryExpr : public AbstractExpr {
public:

    std::unique_ptr<AbstractExpr> left;
    BinaryOperator op;
    std::unique_ptr<AbstractExpr> right;

    /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
    /// \param left is the left operand of the expression.
    /// \param op is the operator of the expression.
    /// \param right is the right operand of the expression.
    BinaryExpr(std::unique_ptr<AbstractExpr> left, const BinaryOperator &op, std::unique_ptr<AbstractExpr> right);

    json toJson() const;

    BinaryOperator getOp() const;


    AbstractExpr *getRight() const;

    AbstractExpr *getLeft() const;
};


#endif //MASTER_THESIS_CODE_BINARYEXPR_H
