#include "Datatypes.h"

Node *Datatype::createClonedNode(bool keepOriginalUniqueNodeId) {
    return new Datatype(this->getType());
}

Datatype::Datatype(const std::string &type) {
    static const std::unordered_map<std::string, types> string_to_types = {
            {"int",    types::Int},
            {"float",  types::Float},
            {"string", types::String},
            {"bool",   types::Bool}};
    auto result = string_to_types.find(type);
    if (result == string_to_types.end()) {
        throw std::invalid_argument(
                "Unsupported datatype given: " + type + ". See the supported datatypes in Datatypes.h.");
    }
    val = result->second;
}

std::string Datatype::enumToString(const types identifiers) {
    static const std::map<types, std::string> types_to_string = {
            {types::Int,    "int"},
            {types::Float,  "float"},
            {types::String, "string"},
            {types::Bool,   "bool"}};
    return types_to_string.find(identifiers)->second;
}

Datatype::operator std::string() const {
    return enumToString(val);
}

Datatype::operator types() const {
    return val;
}

std::string Datatype::toString() const {
    return enumToString(val);
}

bool Datatype::operator==(const Datatype &rhs) const {
    return val == rhs.val &&
           isEncrypted == rhs.isEncrypted;
}

bool Datatype::operator!=(const Datatype &rhs) const {
    return !(rhs == *this);
}

types Datatype::getType() const {
    return val;
}
