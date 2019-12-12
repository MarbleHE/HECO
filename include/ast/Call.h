#ifndef MASTER_THESIS_CODE_CALL_H
#define MASTER_THESIS_CODE_CALL_H


#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "FunctionParameter.h"
#include "AbstractStatement.h"

class Call : public AbstractExpr, public AbstractStatement {
    AbstractExpr *callee; // any expression that evaluates to a function or a Function
    std::vector<FunctionParameter> arguments;
public:
    Call(AbstractExpr *callee, const std::vector<FunctionParameter> &arguments);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

};


#endif //MASTER_THESIS_CODE_CALL_H
