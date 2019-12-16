#ifndef MASTER_THESIS_CODE_UNARYEXPR_H
#define MASTER_THESIS_CODE_UNARYEXPR_H


#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr, public Node {
private:
    Operator *op;
    AbstractExpr *right;

public:
    UnaryExpr(OpSymb::UnaryOp op, AbstractExpr *right);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    Operator *getOp() const;

    AbstractExpr *getRight() const;

    std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_UNARYEXPR_H
