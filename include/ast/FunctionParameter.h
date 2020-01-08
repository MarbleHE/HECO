#ifndef MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
#define MASTER_THESIS_CODE_FUNCTIONPARAMETER_H

#include "Variable.h"
#include <string>

class FunctionParameter : public AbstractExpr {
 private:
  std::string datatype;
  AbstractExpr* value;

 public:
  FunctionParameter(std::string datatype, AbstractExpr* value);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] const std::string &getDatatype() const;

  [[nodiscard]] AbstractExpr* getValue() const;
};

/// Defines the JSON representation to be used for vector<FunctionParameter> objects.
void to_json(json &j, const FunctionParameter &funcParam);

/// Defines the JSON representation to be used for vector<FunctionParameter *> objects.
void to_json(json &j, const FunctionParameter* funcParam);

#endif //MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
