
#ifndef MASTER_THESIS_CODE_CALL_H
#define MASTER_THESIS_CODE_CALL_H


#include <string>
#include <vector>
#include "ExpressionStmt.h"

class Call : public AbstractExpr {
    AbstractExpr callee; // any expression that evaluates to a function or a Function
    std::vector<AbstractExpr> arguments;
public:
    Call(const AbstractExpr &callee, const std::vector<AbstractExpr> &arguments);
};


#endif //MASTER_THESIS_CODE_CALL_H
