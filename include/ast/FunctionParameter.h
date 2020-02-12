#ifndef MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
#define MASTER_THESIS_CODE_FUNCTIONPARAMETER_H

#include "Variable.h"
#include <string>
#include "Datatypes.h"

class FunctionParameter : public AbstractExpr {
 public:
  FunctionParameter(Datatype* datatype, AbstractExpr* value);

  /// Helper constructor for convenience.
  /// \param datatype A valid datatype according to TYPES in Datatype.h
  /// \param value The value of the function parameter.
  FunctionParameter(std::string datatype, AbstractExpr* value);

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] Datatype* getDatatype() const;

  [[nodiscard]] AbstractExpr* getValue() const;

  void accept(Visitor &v) override;

 private:
  Datatype* datatype;
  AbstractExpr* value;
};

/// Defines the JSON representation to be used for vector<FunctionParameter> objects.
void to_json(json &j, const FunctionParameter &funcParam);

/// Defines the JSON representation to be used for vector<FunctionParameter *> objects.
void to_json(json &j, const FunctionParameter* funcParam);

#endif //MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
