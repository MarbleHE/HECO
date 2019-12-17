#ifndef MASTER_THESIS_CODE_IF_H
#define MASTER_THESIS_CODE_IF_H


#include "AbstractStatement.h"
#include "AbstractExpr.h"

class If : public AbstractStatement, public Node {
private:
    AbstractExpr *condition;
    AbstractStatement *thenBranch;
    AbstractStatement *elseBranch;

public:
    If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch);

    If(AbstractExpr *condition, AbstractStatement *thenBranch);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;

    AbstractExpr *getCondition() const;

    AbstractStatement *getThenBranch() const;

    AbstractStatement *getElseBranch() const;

    virtual ~If();
};


#endif //MASTER_THESIS_CODE_IF_H
