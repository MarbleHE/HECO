
#include <vector>
#include <FunctionParameter.h>
#include "CallExternal.h"

CallExternal::CallExternal(const std::string &functionName, const std::vector<FunctionParameter> &arguments)
        : functionName(functionName), arguments(arguments) {}

CallExternal::CallExternal(const std::string &functionName) {

}


json CallExternal::toJson() const {
    json j;
    j["type"] = "CallExternal";
    j["functionName"] = this->functionName;
    j["arguments"] = this->arguments;
    return j;
}
