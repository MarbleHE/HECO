#ifndef AST_OPTIMIZER_INCLUDE_CALL_H
#define AST_OPTIMIZER_INCLUDE_CALL_H

#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "FunctionParameter.h"
#include "AbstractStatement.h"

class Call : public AbstractExpr, public AbstractStatement {
private:
    Function *func{nullptr};
    std::vector<FunctionParameter *> arguments;

    AbstractNode *createClonedNode(bool keepOriginalUniqueNodeId) override;

public:
    Call(std::vector<FunctionParameter *> arguments, Function *func);

    explicit Call(Function *func);

    ~Call() override;

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] const std::vector<FunctionParameter *> &getArguments() const;

    [[nodiscard]] std::string getNodeName() const override;

    [[nodiscard]] Function *getFunc() const;
};

#endif //AST_OPTIMIZER_INCLUDE_CALL_H
