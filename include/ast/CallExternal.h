#ifndef MASTER_THESIS_CODE_CALLEXTERNAL_H
#define MASTER_THESIS_CODE_CALLEXTERNAL_H


#include "AbstractStatement.h"

class CallExternal : public AbstractExpr, public AbstractStatement, public Node {
private:
    std::string functionName;
    std::vector<FunctionParameter> *arguments;

public:
    explicit CallExternal(std::string functionName);

    CallExternal(std::string functionName, std::vector<FunctionParameter> *arguments);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    const std::string &getFunctionName() const;

    std::vector<FunctionParameter> *getArguments() const;

    std::string getNodeName() const override;
};


#endif //MASTER_THESIS_CODE_CALLEXTERNAL_H
