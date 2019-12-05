//
// Created by Patrick Jattke on 04.12.19.
//

#ifndef MASTER_THESIS_CODE_IF_H
#define MASTER_THESIS_CODE_IF_H


#include "AbstractStatement.h"
#include "ExpressionStmt.h"

class If : public AbstractStatement {
    AbstractExpr* condition;
    AbstractStatement* thenBranch;
    AbstractStatement* elseBranch;
public:
    If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch);
};


#endif //MASTER_THESIS_CODE_IF_H
