#ifndef AST_OPTIMIZER_INCLUDE_IF_H
#define AST_OPTIMIZER_INCLUDE_IF_H

#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include <string>

class If : public AbstractStatement {
private:
    AbstractExpr *condition;
    AbstractStatement *thenBranch;
    AbstractStatement *elseBranch;

public:
    If(AbstractExpr *condition, AbstractStatement *thenBranch);

    If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch);

    ~If() override;

    If *clone(bool keepOriginalUniqueNodeId) override;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    [[nodiscard]] AbstractExpr *getCondition() const;

    [[nodiscard]] AbstractStatement *getThenBranch() const;

    [[nodiscard]] AbstractStatement *getElseBranch() const;

};

#endif //AST_OPTIMIZER_INCLUDE_IF_H
