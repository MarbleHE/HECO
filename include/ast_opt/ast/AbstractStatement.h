#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTSTATEMENT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTSTATEMENT_H_

#include <string>
#include <nlohmann/json.hpp>
#include "ast_opt/visitor/Visitor.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/AbstractBinaryExpr.h"

using json = nlohmann::json;

class AbstractStatement : public AbstractNode {
 public:
  [[nodiscard]] json toJson() const override;

  virtual AbstractBinaryExpr *contains(AbstractBinaryExpr *aexpTemplate, ArithmeticExpr *excludedSubtree);

  virtual std::string getVarTargetIdentifier() const;

  virtual bool isEqual(AbstractStatement *as);
};

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj);

/// JSON representation to be used for vector<AbstractStatement> objects.
void to_json(json &j, const AbstractStatement &absStat);

void to_json(json &j, const AbstractStatement *absStat);

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTSTATEMENT_H_
