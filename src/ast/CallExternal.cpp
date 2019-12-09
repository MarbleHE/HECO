
#include <vector>
#include <FunctionParameter.h>
#include "CallExternal.h"

CallExternal::CallExternal(const std::string &functionName, const std::vector<FunctionParameter> &arguments)
        : functionName(functionName), arguments(arguments) {}

CallExternal::CallExternal(const std::string &functionName) {

}
