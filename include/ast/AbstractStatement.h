#ifndef AST_OPTIMIZER_INCLUDE_ABSTRACTSTATEMENT_H
#define AST_OPTIMIZER_INCLUDE_ABSTRACTSTATEMENT_H

#include <string>
#include <nlohmann/json.hpp>
#include "Visitor.h"
#include "Node.h"

using json = nlohmann::json;

class AbstractStatement : public Node {
 public:
  [[nodiscard]] std::string toString() const override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  virtual BinaryExpr* contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree);

  virtual std::string getVarTargetIdentifier();

  virtual bool isEqual(AbstractStatement* as);
};

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj);

/// JSON representation to be used for vector<AbstractStatement> objects.
void to_json(json &j, const AbstractStatement &absStat);

void to_json(json &j, const AbstractStatement* absStat);

#endif //AST_OPTIMIZER_INCLUDE_ABSTRACTSTATEMENT_H
