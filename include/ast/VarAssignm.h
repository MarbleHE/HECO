#ifndef MASTER_THESIS_CODE_VARASSIGNM_H
#define MASTER_THESIS_CODE_VARASSIGNM_H


#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarAssignm : public AbstractStatement {
    std::string identifier;
    AbstractExpr *value;
public:
    VarAssignm(const std::string &identifier, AbstractExpr *value);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

};

#endif //MASTER_THESIS_CODE_VARASSIGNM_H
