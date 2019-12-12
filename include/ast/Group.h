#ifndef MASTER_THESIS_CODE_GROUP_H
#define MASTER_THESIS_CODE_GROUP_H


#include "AbstractExpr.h"

class Group : public AbstractExpr {
    std::unique_ptr<AbstractExpr> expr;
public:
    Group(std::unique_ptr<AbstractExpr> expr);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_GROUP_H
