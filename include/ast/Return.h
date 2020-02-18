#ifndef AST_OPTIMIZER_INCLUDE_RETURN_H
#define AST_OPTIMIZER_INCLUDE_RETURN_H

#include <string>
#include <vector>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class Return : public AbstractStatement {
public:
    Return();

    explicit Return(AbstractExpr *returnValue);

    explicit Return(std::vector<AbstractExpr *> returnValues);

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    ~Return() override;

    [[nodiscard]] std::vector<AbstractExpr *> getReturnExpressions() const;

    Return *clone(bool keepOriginalUniqueNodeId) override;

    void setAttributes(std::vector<AbstractExpr *> returnExpr);

protected:
    int getMaxNumberChildren() override;

    bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_RETURN_H
