#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTIONPARAMETER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTIONPARAMETER_H_

#include "Variable.h"
#include <string>
#include "Datatype.h"

class FunctionParameter : public AbstractExpr {
 public:
  FunctionParameter(Datatype *datatype, AbstractExpr *value);

  /// Helper constructor for keeping downwards compatibility with earlier interface.
  /// \deprecated This constructor should not be used anymore, use the one requiring a Datatype instead.
  /// \param datatypeEnumString A valid datatype according to types in Datatype.h
  /// \param value The value of the function parameter.
  FunctionParameter(const std::string &datatypeEnumString, AbstractExpr *value);

  FunctionParameter *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] std::string getNodeType() const override;

  [[nodiscard]] Datatype *getDatatype() const;

  [[nodiscard]] AbstractExpr *getValue() const;

  void accept(Visitor &v) override;

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;

  void setAttributes(Datatype *datatype, AbstractExpr *value);

  bool operator==(const FunctionParameter &rhs) const;

  bool operator!=(const FunctionParameter &rhs) const;

  std::vector<std::string> getVariableIdentifiers() override;

  std::vector<Variable *> getVariables() override;

  [[nodiscard]] std::string toString(bool printChildren) const override;
};

/// Defines the JSON representation to be used for vector<FunctionParameter> objects.
void to_json(json &j, const FunctionParameter &funcParam);

/// Defines the JSON representation to be used for vector<FunctionParameter *> objects.
void to_json(json &j, const FunctionParameter *funcParam);

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTIONPARAMETER_H_
