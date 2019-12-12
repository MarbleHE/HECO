#ifndef MASTER_THESIS_CODE_WHILE_H
#define MASTER_THESIS_CODE_WHILE_H


#include "AbstractStatement.h"
#include "AbstractExpr.h"

class While : public AbstractStatement {
    AbstractExpr *condition;
    AbstractStatement *body;
public:
    While(AbstractExpr *condition, AbstractStatement *body);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_WHILE_H
