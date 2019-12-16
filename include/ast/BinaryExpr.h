#ifndef MASTER_THESIS_CODE_BINARYEXPR_H
#define MASTER_THESIS_CODE_BINARYEXPR_H

#include "Operator.h"
#include "AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"

class BinaryExpr : public AbstractExpr, public Node {
protected:
    AbstractExpr *left;
    Operator *op;
    AbstractExpr *right;

public:
    /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
    /// \param left is the left operand of the expression.
    /// \param op is the operator of the expression.
    /// \param right is the right operand of the expression.
    BinaryExpr(AbstractExpr *left, OpSymb::BinaryOp op, AbstractExpr *right);

    template<typename T1, typename T2>
    BinaryExpr(T1 left, OpSymb::BinaryOp op, T2 right) {
        this->left = createParam(left);
        this->op = new Operator(op);
        this->right = createParam(right);
    }

    json toJson() const override;

    AbstractExpr *getLeft() const;

    Operator &getOp() const;

    AbstractExpr *getRight() const;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_BINARYEXPR_H
