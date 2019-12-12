#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"

class BinaryExpr : public AbstractExpr {
protected:
    AbstractExpr *left;
    BinaryOperator op;
    AbstractExpr *right;

public:
    /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
    /// \param left is the left operand of the expression.
    /// \param op is the operator of the expression.
    /// \param right is the right operand of the expression.
    BinaryExpr(AbstractExpr *left, BinaryOperator op, AbstractExpr *right);

    // helper constructors that automatically instantiate the underlying objects for common variations
    BinaryExpr(const std::string &variableLeft, BinaryOperator op, const std::string &variableRight);

    BinaryExpr(int literalIntLeft, BinaryOperator op, const std::string &variableRight);

    BinaryExpr(const std::string &variableLeft, BinaryOperator op, int literalIntRight);

    BinaryExpr(int literalIntLeft, BinaryOperator op, int literalIntRight);

    json toJson() const override;

    AbstractExpr *getLeft() const;

    BinaryOperator getOp() const;

    AbstractExpr *getRight() const;

    virtual void accept(Visitor &v) override;

};


#endif //MASTER_THESIS_CODE_BINARYEXPR_H
