#ifndef MASTER_THESIS_CODE_AST_H
#define MASTER_THESIS_CODE_AST_H

#include <map>
#include <string>
#include "AbstractStatement.h"
#include "../visitor/Visitor.h"

class Ast {
 private:
  AbstractStatement* rootNode;
  std::map<std::string, Literal*> variablesValues;

 public:
  Ast();

  explicit Ast(AbstractStatement* rootNode);

  ~Ast();

  AbstractStatement* setRootNode(AbstractStatement* node);

  [[nodiscard]] AbstractStatement* getRootNode() const;

  virtual void accept(Visitor &v);

  bool hasVarValue(Variable* var);

  Literal* getVarValue(const std::string &variableIdentifier);

  void updateVarValue(const std::string &variableIdentifier, Literal* newValue);

  Literal* evaluate(std::map<std::string, Literal*> &paramValues, bool printResult);
};

#endif //MASTER_THESIS_CODE_AST_H
