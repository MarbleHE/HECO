#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTION_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTION_H_

#include <string>
#include <vector>
#include "AbstractStatement.h"
#include "ParameterList.h"
#include "FunctionParameter.h"
#include "VarDecl.h"

/// Function has two children: First, a ParameterList and then a Block with the actual function Body
class Function : public AbstractStatement {
 private:
  std::string name;

 public:
  Function *clone(bool keepOriginalUniqueNodeId) const override;

  [[nodiscard]] const std::string &getName() const;

  [[nodiscard]] std::vector<FunctionParameter*> getParameters() const;

  [[nodiscard]] std::vector<AbstractStatement*> getBodyStatements() const;

  Function(std::string name, Block *pt);

  Function(std::string functionName, ParameterList *functionParameters,
           Block *functionStatements);

  Function(std::string functionName, std::vector<FunctionParameter*> functionParameters,
           std::vector<AbstractStatement*> functionStatements);

  explicit Function(std::string name);

  void addParameter(FunctionParameter *param);

  void addStatement(AbstractStatement *pDecl);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  [[nodiscard]] ParameterList *getParameterList() const;

  [[nodiscard]] Block *getBody() const;

  void setParameterList(ParameterList *paramsVec);

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;

  [[nodiscard]] std::string toString(bool printChildren) const override;
};

/// Defines the JSON representation to be used for vector<Function> objects.
void to_json(json &j, const Function &func);

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FUNCTION_H_
