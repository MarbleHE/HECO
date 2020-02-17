#ifndef AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPES_H_
#define AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPES_H_

#include <string>
#include <map>
#include "AbstractNode.h"

enum class TYPES {
    INT, FLOAT, STRING, BOOL
};

class Datatype : public AbstractNode {
private:
    AbstractNode *createClonedNode(bool keepOriginalUniqueNodeId) override {
        return new Datatype(this->getType());
    }

    TYPES val;
    bool isEncrypted = false;

public:
    explicit Datatype(TYPES di) : val(di) {}

    explicit Datatype(TYPES di, bool isEncrypted) : val(di), isEncrypted(isEncrypted) {}

    explicit Datatype(const std::string &type) {
        static const std::unordered_map<std::string, TYPES> string_to_types = {
                {"int",    TYPES::INT},
                {"float",  TYPES::FLOAT},
                {"string", TYPES::STRING},
                {"bool",   TYPES::BOOL}};
        auto result = string_to_types.find(type);
        if (result == string_to_types.end()) {
            throw std::invalid_argument(
                    "Unsupported datatype given: " + type + ". See the supported datatypes in Datatypes.h.");
        }
        val = result->second;
    }

    static std::string enum_to_string(const TYPES identifiers) {
        static const std::map<TYPES, std::string> types_to_string = {
                {TYPES::INT,    "int"},
                {TYPES::FLOAT,  "float"},
                {TYPES::STRING, "string"},
                {TYPES::BOOL,   "bool"}};
        return types_to_string.find(identifiers)->second;
    }

    explicit operator std::string() const {
        return enum_to_string(val);
    }

    explicit operator TYPES() const {
        return val;
    }

    [[nodiscard]] std::string toString() const override {
        return enum_to_string(val);
    }

    bool operator==(const Datatype &rhs) const {
        return val == rhs.val &&
               isEncrypted == rhs.isEncrypted;
    }

    bool operator!=(const Datatype &rhs) const {
        return !(rhs == *this);
    }

    [[nodiscard]] TYPES getType() const {
        return val;
    }
};

#endif //AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPES_H_
