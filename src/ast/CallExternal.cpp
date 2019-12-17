#include <utility>
#include <vector>
#include <FunctionParameter.h>
#include "CallExternal.h"


json CallExternal::toJson() const {
    json j = {
            {"type",         getNodeName()},
            {"functionName", this->functionName},
    };
    if (this->arguments != nullptr) {
        j["arguments"] = *(this->arguments);
    }
    return j;
}

CallExternal::CallExternal(std::string functionName, std::vector<FunctionParameter> *arguments) : functionName(
        std::move(functionName)) {
    this->arguments = arguments;
}

CallExternal::CallExternal(std::string functionName) : functionName(functionName) {
    this->arguments = nullptr;
}

void CallExternal::accept(Visitor &v) {
    v.visit(*this);
}

const std::string &CallExternal::getFunctionName() const {
    return functionName;
}

std::vector<FunctionParameter> *CallExternal::getArguments() const {
    return arguments;
}

std::string CallExternal::getNodeName() const {
    return "CallExternal";
}
