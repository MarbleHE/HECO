//
// Created by Patrick Jattke on 04.12.19.
//

#ifndef MASTER_THESIS_CODE_RETURN_H
#define MASTER_THESIS_CODE_RETURN_H


#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class Return : public AbstractStatement {
public:
    std::unique_ptr<AbstractExpr> value;

    Return(std::unique_ptr<AbstractExpr> value);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_RETURN_H
