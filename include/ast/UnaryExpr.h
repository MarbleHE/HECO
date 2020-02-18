#ifndef AST_OPTIMIZER_INCLUDE_UNARYEXPR_H
#define AST_OPTIMIZER_INCLUDE_UNARYEXPR_H

#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr {
public:
    UnaryExpr(OpSymb::UnaryOp op, AbstractExpr *right);

    UnaryExpr *clone(bool keepOriginalUniqueNodeId) override;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] Operator *getOp() const;

    [[nodiscard]] AbstractExpr *getRight() const;

    [[nodiscard]] std::string getNodeName() const override;

    ~UnaryExpr() override;

    void setAttributes(OpSymb::UnaryOp op, AbstractExpr *expr);

protected:
    bool supportsCircuitMode() override;

    int getMaxNumberChildren() override;

};

#endif //AST_OPTIMIZER_INCLUDE_UNARYEXPR_H
