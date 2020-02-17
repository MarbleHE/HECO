#ifndef AST_OPTIMIZER_INCLUDE_FUNCTIONPARAMETER_H
#define AST_OPTIMIZER_INCLUDE_FUNCTIONPARAMETER_H

#include "Variable.h"
#include <string>
#include "Datatypes.h"

class FunctionParameter : public AbstractExpr {
private:
    Node *createClonedNode(bool keepOriginalUniqueNodeId) override;

public:
    FunctionParameter(Datatype *datatype, AbstractExpr *value);

    /// Helper constructor for keeping downwards compatibility with earlier interface.
    /// \deprecated This constructor should not be used anymore, use the one requiring a Datatype instead.
    /// \param datatypeEnumString A valid datatype according to types in Datatype.h
    /// \param value The value of the function parameter.
    FunctionParameter(const std::string &datatypeEnumString, AbstractExpr *value);

    [[nodiscard]] json toJson() const override;

    [[nodiscard]] std::string getNodeName() const override;

    [[nodiscard]] Datatype *getDatatype() const;

    [[nodiscard]] AbstractExpr *getValue() const;

    void accept(Visitor &v) override;

    int getMaxNumberChildren() override;

    bool supportsCircuitMode() override;

    void setAttributes(Datatype *datatype, AbstractExpr *value);

    bool operator==(const FunctionParameter &rhs) const;

    bool operator!=(const FunctionParameter &rhs) const;
};

/// Defines the JSON representation to be used for vector<FunctionParameter> objects.
void to_json(json &j, const FunctionParameter &funcParam);

/// Defines the JSON representation to be used for vector<FunctionParameter *> objects.
void to_json(json &j, const FunctionParameter *funcParam);

#endif //AST_OPTIMIZER_INCLUDE_FUNCTIONPARAMETER_H
