#ifndef MASTER_THESIS_CODE_CALL_H
#define MASTER_THESIS_CODE_CALL_H


#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "FunctionParameter.h"
#include "AbstractStatement.h"

class Call : public AbstractExpr, public AbstractStatement, public Node {
private:
    AbstractExpr *callee; // any expression that evaluates to a function or a Function
    std::vector<FunctionParameter *> arguments;
public:
    Call(AbstractExpr *callee, std::vector<FunctionParameter *> arguments);

    explicit Call(AbstractExpr *callee);

    ~Call();

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] AbstractExpr *getCallee() const;

    [[nodiscard]] const std::vector<FunctionParameter *> &getArguments() const;

    [[nodiscard]] std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_CALL_H
