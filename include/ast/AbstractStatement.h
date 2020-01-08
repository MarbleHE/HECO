#ifndef MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
#define MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H

#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"
#include "Node.h"
#include "Ast.h"

using json = nlohmann::json;

class AbstractStatement : public Node {
 public:
  virtual ~AbstractStatement() = default;

  [[nodiscard]] virtual std::string toString() const;

  [[nodiscard]] virtual json toJson() const;

  virtual void accept(Visitor &v);

  virtual BinaryExpr* contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree);

  virtual std::string getVarTargetIdentifier();

  virtual bool isEqual(AbstractStatement* as);

  virtual Literal* evaluate(Ast &ast);
};

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj);

/// JSON representation to be used for vector<AbstractStatement> objects.
void to_json(json &j, const AbstractStatement &absStat);

void to_json(json &j, const AbstractStatement* absStat);

#endif //MASTER_THESIS_CODE_ABSTRACTSTATEMENT_H
