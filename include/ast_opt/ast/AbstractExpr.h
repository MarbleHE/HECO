#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTEXPR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTEXPR_H_

#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "ast_opt/visitor/Visitor.h"
#include "ast_opt/ast/Ast.h"
#include "ast_opt/ast/AbstractNode.h"

using json = nlohmann::json;

class AbstractBinaryExpr;

class AbstractExpr : public AbstractNode {
 protected:
  static LiteralInt *createParam(int i);

  static LiteralBool *createParam(bool b);

  static LiteralString *createParam(const char *str);

  static LiteralFloat *createParam(float f);

  static AbstractExpr *createParam(AbstractExpr *abstractExpr);

  static AbstractNode *createParam(AbstractNode *node);

 public:
  [[nodiscard]] json toJson() const override;

  virtual AbstractBinaryExpr *contains(AbstractBinaryExpr *aexpTemplate, AbstractExpr *excludedSubtree);

  virtual int countByTemplate(AbstractExpr *abstractExpr);

  virtual std::vector<std::string> getVariableIdentifiers() = 0;

  virtual std::vector<Variable*> getVariables()  = 0;

  virtual bool contains(Variable *var);

  virtual bool isEqual(AbstractExpr *other);

  AbstractExpr() = default;
};

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj);

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTEXPR_H_
