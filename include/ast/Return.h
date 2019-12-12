#ifndef MASTER_THESIS_CODE_RETURN_H
#define MASTER_THESIS_CODE_RETURN_H


#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class Return : public AbstractStatement {
public:
    AbstractExpr *value;

    Return(AbstractExpr *value);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_RETURN_H
