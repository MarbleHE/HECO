#ifndef MASTER_THESIS_CODE_VARASSIGNM_H
#define MASTER_THESIS_CODE_VARASSIGNM_H


#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarAssignm : public AbstractStatement {
    std::string identifier;
    std::unique_ptr<AbstractExpr> value;
public:
    VarAssignm(const std::string &identifier, std::unique_ptr<AbstractExpr> value);

    json toJson() const;
};

#endif //MASTER_THESIS_CODE_VARASSIGNM_H
