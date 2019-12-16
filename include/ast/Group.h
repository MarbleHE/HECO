#ifndef MASTER_THESIS_CODE_GROUP_H
#define MASTER_THESIS_CODE_GROUP_H


#include "AbstractExpr.h"

class Group : public AbstractExpr, public Node {
private:
    AbstractExpr *expr;
public:

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;

    AbstractExpr *getExpr() const;

    Group(AbstractExpr *expr);
};


#endif //MASTER_THESIS_CODE_GROUP_H
