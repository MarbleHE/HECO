//
// Created by Patrick Jattke on 04.12.19.
//

#ifndef MASTER_THESIS_CODE_RETURN_H
#define MASTER_THESIS_CODE_RETURN_H


#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class Return : public AbstractStatement {
    AbstractExpr* value;
public:
    Return(AbstractExpr *value);
};


#endif //MASTER_THESIS_CODE_RETURN_H
