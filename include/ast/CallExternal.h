#ifndef MASTER_THESIS_CODE_CALLEXTERNAL_H
#define MASTER_THESIS_CODE_CALLEXTERNAL_H


#include "AbstractStatement.h"

class CallExternal : public AbstractExpr, public AbstractStatement {
public:
    std::string functionName;
    std::vector<FunctionParameter> arguments;

    CallExternal(const std::string &functionName);

    CallExternal(const std::string &functionName, const std::vector<FunctionParameter> &arguments);

    json toJson() const;
};


#endif //MASTER_THESIS_CODE_CALLEXTERNAL_H
