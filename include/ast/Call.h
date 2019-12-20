#ifndef MASTER_THESIS_CODE_CALL_H
#define MASTER_THESIS_CODE_CALL_H


#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "FunctionParameter.h"
#include "AbstractStatement.h"

class Call : public AbstractExpr, public AbstractStatement, public Node {
private:
    Function *func;
    std::vector<FunctionParameter *> arguments;
public:
    Call(std::vector<FunctionParameter *> arguments, Function *func);

    Call(Function *func);

    ~Call();

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] const std::vector<FunctionParameter *> &getArguments() const;

    [[nodiscard]] std::string getNodeName() const override;

    Function *getFunc() const;
};


#endif //MASTER_THESIS_CODE_CALL_H
