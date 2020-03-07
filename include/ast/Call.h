#ifndef AST_OPTIMIZER_INCLUDE_AST_CALL_H_
#define AST_OPTIMIZER_INCLUDE_AST_CALL_H_

#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "FunctionParameter.h"
#include "AbstractStatement.h"

class Call : public AbstractExpr {
 public:
  Call(std::vector<FunctionParameter *> parameterValuesForCalledFunction, Function *func);

  explicit Call(Function *func);

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::vector<FunctionParameter *> getArguments() const;

  [[nodiscard]] ParameterList *getParameterList() const;

  [[nodiscard]] std::string getNodeType() const override;

  [[nodiscard]] Function *getFunc() const;

  void setAttributes(std::vector<FunctionParameter *> functionCallParameters, Function *functionToBeCalled);

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_CALL_H_
