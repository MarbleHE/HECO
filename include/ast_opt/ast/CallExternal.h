#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_CALLEXTERNAL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_CALLEXTERNAL_H_

#include "AbstractStatement.h"
#include <vector>
#include <string>

class CallExternal : public AbstractExpr {
 private:
  std::string functionName;
  std::vector<FunctionParameter *> arguments;

 public:
  explicit CallExternal(std::string functionName);

  CallExternal(std::string functionName, std::vector<FunctionParameter *> arguments);

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] const std::string &getFunctionName() const;

  [[nodiscard]] const std::vector<FunctionParameter *> &getArguments() const;

  [[nodiscard]] std::string getNodeType() const override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_CALLEXTERNAL_H_
