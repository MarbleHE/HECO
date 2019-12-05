#ifndef MASTER_THESIS_CODE_VARASSIGNM_H
#define MASTER_THESIS_CODE_VARASSIGNM_H


#include <string>
#include "ExpressionStmt.h"

class VarAssignm : public AbstractStatement {
    std::string identifier;
    AbstractExpr *value;
public:
    VarAssignm(const std::string &identifier, AbstractExpr *value);

};

#endif //MASTER_THESIS_CODE_VARASSIGNM_H
