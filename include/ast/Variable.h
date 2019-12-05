
#ifndef MASTER_THESIS_CODE_VARIABLE_H
#define MASTER_THESIS_CODE_VARIABLE_H


#include <string>
#include "ExpressionStmt.h"

class Variable : public AbstractExpr {
    std::string identifier;
public:
    Variable(const std::string &identifier);
};


#endif //MASTER_THESIS_CODE_VARIABLE_H
