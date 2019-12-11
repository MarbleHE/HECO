//
// Created by Patrick Jattke on 04.12.19.
//

#ifndef MASTER_THESIS_CODE_IF_H
#define MASTER_THESIS_CODE_IF_H


#include "AbstractStatement.h"
#include "AbstractExpr.h"

class If : public AbstractStatement {
public:


    std::unique_ptr<AbstractExpr> condition;
    std::unique_ptr<AbstractStatement> thenBranch;
    std::unique_ptr<AbstractStatement> elseBranch;

    If(std::unique_ptr<AbstractExpr> condition, std::unique_ptr<AbstractStatement> thenBranch);

    If(std::unique_ptr<AbstractExpr> condition, std::unique_ptr<AbstractStatement> thenBranch,
       std::unique_ptr<AbstractStatement> elseBranch);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_IF_H
