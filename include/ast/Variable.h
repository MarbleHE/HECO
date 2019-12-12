#ifndef MASTER_THESIS_CODE_VARIABLE_H
#define MASTER_THESIS_CODE_VARIABLE_H


#include <string>
#include "AbstractExpr.h"

class Variable : public AbstractExpr {
public:
    Variable(const std::string &identifier);

    std::string identifier;

    json toJson() const override;

    virtual void accept(Visitor &v) override;

};


#endif //MASTER_THESIS_CODE_VARIABLE_H
