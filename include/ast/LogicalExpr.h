#ifndef MASTER_THESIS_CODE_LOGICALEXPR_H
#define MASTER_THESIS_CODE_LOGICALEXPR_H


#include "Operator.h"
#include "AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"


class LogicalExpr : public AbstractExpr, public Node {
private:
    AbstractExpr *left;
    Operator *op;
    AbstractExpr *right;

public:
    LogicalExpr(AbstractExpr *left, OpSymb::LogCompOp op, AbstractExpr *right);

    template<typename T1, typename T2>
    LogicalExpr(T1 left, OpSymb::LogCompOp op, T2 right) {
        this->left = createParam(left);
        this->op = new Operator(op);
        this->right = createParam(right);
    }

    AbstractExpr *getLeft() const;

    Operator *getOp() const;

    AbstractExpr *getRight() const;

    json toJson() const
    override;

    virtual void accept(Visitor &v)
    override;

    std::string getNodeName() const override;

};


#endif //MASTER_THESIS_CODE_LOGICALEXPR_H
