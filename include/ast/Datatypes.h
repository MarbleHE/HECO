#ifndef AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPES_H_
#define AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPES_H_

#include <string>
#include <map>
#include "Node.h"

enum class types {
    Int, Float, String, Bool
};

class Datatype : public Node {
private:
    Node *createClonedNode(bool keepOriginalUniqueNodeId) override;

    types val;
    bool isEncrypted = false;

public:
    explicit Datatype(types di) : val(di) {}

    explicit Datatype(types di, bool isEncrypted) : val(di), isEncrypted(isEncrypted) {}

    explicit Datatype(const std::string &type);

    static std::string enumToString(types identifiers);

    explicit operator std::string() const;

    explicit operator types() const;

    [[nodiscard]] std::string toString() const override;

    bool operator==(const Datatype &rhs) const;

    bool operator!=(const Datatype &rhs) const;

    [[nodiscard]] types getType() const;
};

#endif //AST_OPTIMIZER_INCLUDE_INCLUDE_UTILITIES_DATATYPES_H_
