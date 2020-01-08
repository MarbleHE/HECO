#ifndef MASTER_THESIS_CODE_FUNCTION_H
#define MASTER_THESIS_CODE_FUNCTION_H

#include <string>
#include <vector>
#include "AbstractStatement.h"
#include "FunctionParameter.h"
#include "VarDecl.h"

class Function : public AbstractStatement {
 private:
  std::string name;
  std::vector<FunctionParameter> params;
  std::vector<AbstractStatement*> body;

 public:
  Function() = default;

  [[nodiscard]] const std::string &getName() const;

  [[nodiscard]] const std::vector<FunctionParameter> &getParams() const;

  [[nodiscard]] const std::vector<AbstractStatement*> &getBody() const;

  /// Copy constructor
  /// \param func The function to be copied.
  Function(const Function &func);

  Function(std::string name, std::vector<AbstractStatement*> bodyStatements);

  Function(std::string name, std::vector<FunctionParameter> params,
           std::vector<AbstractStatement*> body);

  explicit Function(std::string name);

  void addParameter(FunctionParameter* param);

  void addStatement(AbstractStatement* pDecl);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  void setParams(std::vector<FunctionParameter>* paramsVec);

  Literal* evaluate(Ast &ast) override;
};

/// Defines the JSON representation to be used for vector<Function> objects.
void to_json(json &j, const Function &func);

#endif //MASTER_THESIS_CODE_FUNCTION_H
