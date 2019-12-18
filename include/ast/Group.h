#ifndef MASTER_THESIS_CODE_GROUP_H
#define MASTER_THESIS_CODE_GROUP_H


#include "AbstractExpr.h"

class Group : public AbstractExpr, public Node {
private:
    AbstractExpr *expr;
public:

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    [[nodiscard]] AbstractExpr *getExpr() const;

    explicit Group(AbstractExpr *expr);

    ~Group();
};


#endif //MASTER_THESIS_CODE_GROUP_H
