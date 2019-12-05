
#ifndef MASTER_THESIS_CODE_UNARYEXPR_H
#define MASTER_THESIS_CODE_UNARYEXPR_H


#include <string>
#include "ExpressionStmt.h"

class UnaryExpr : AbstractExpr {
    std::string op;
    AbstractExpr right;
public:
    UnaryExpr(const std::string &op, const AbstractExpr &right);
};


#endif //MASTER_THESIS_CODE_UNARYEXPR_H
