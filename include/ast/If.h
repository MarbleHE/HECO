#ifndef MASTER_THESIS_CODE_IF_H
#define MASTER_THESIS_CODE_IF_H


#include "AbstractStatement.h"
#include "AbstractExpr.h"

class If : public AbstractStatement {
public:


    AbstractExpr *condition;
    AbstractStatement *thenBranch;
    AbstractStatement *elseBranch;

    If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch);

    If(AbstractExpr *condition, AbstractStatement *thenBranch);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_IF_H
