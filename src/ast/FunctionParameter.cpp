
#include "../../include/ast/FunctionParameter.h"

#include <utility>

FunctionParameter::FunctionParameter(const std::string &identifier,
                                     std::string datatype) : Variable(identifier), datatype(std::move(datatype)) {}
