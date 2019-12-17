#ifndef MASTER_THESIS_CODE_WHILE_H
#define MASTER_THESIS_CODE_WHILE_H


#include "AbstractStatement.h"
#include "AbstractExpr.h"

class While : public AbstractStatement, public Node {
private:
    AbstractExpr *condition;
    AbstractStatement *body;
public:
    While(AbstractExpr *condition, AbstractStatement *body);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    AbstractExpr *getCondition() const;

    AbstractStatement *getBody() const;

    std::string getNodeName() const override;

    virtual ~While();
};


#endif //MASTER_THESIS_CODE_WHILE_H
