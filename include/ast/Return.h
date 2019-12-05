//
// Created by Patrick Jattke on 04.12.19.
//

#ifndef MASTER_THESIS_CODE_RETURN_H
#define MASTER_THESIS_CODE_RETURN_H


#include <string>
#include "ExpressionStmt.h"
#include "AbstractStatement.h"

class Return : public AbstractStatement {
    AbstractExpr* value;
public:
    Return(AbstractExpr *value);
};


#endif //MASTER_THESIS_CODE_RETURN_H
