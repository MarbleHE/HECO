
#include <utility>
#include <vector>
#include <FunctionParameter.h>
#include "CallExternal.h"




json CallExternal::toJson() const {
    json j = {
            {"type",         "CallExternal"},
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

}
