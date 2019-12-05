
#ifndef MASTER_THESIS_CODE_GROUP_H
#define MASTER_THESIS_CODE_GROUP_H


#include "ExpressionStmt.h"

class Group : public AbstractExpr {
    AbstractExpr* expr;
public:
    Group(AbstractExpr *expr);
};


#endif //MASTER_THESIS_CODE_GROUP_H
