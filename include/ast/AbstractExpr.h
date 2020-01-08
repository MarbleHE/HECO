#ifndef MASTER_THESIS_CODE_ABSTRACTEXPR_H
#define MASTER_THESIS_CODE_ABSTRACTEXPR_H

#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"
#include "Node.h"
#include "Ast.h"

using json = nlohmann::json;

class AbstractExpr : public Node {
 private:
  static LiteralInt* createParam(int i);

  static LiteralBool* createParam(bool b);

  static LiteralString* createParam(const char* str);

  static AbstractExpr* createParam(AbstractExpr* abstractExpr);

 public:
  virtual ~AbstractExpr() = default;

  [[nodiscard]] virtual std::string toString() const;

  [[nodiscard]] virtual json toJson() const;

  virtual void accept(Visitor &v);

  virtual BinaryExpr* contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree);

  virtual bool contains(Variable* var);

  virtual bool isEqual(AbstractExpr* other);

  virtual Literal* evaluate(Ast &ast);
};

/// JSON representation to be used for vector<AbstractExpr> objects.
std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj);

#endif //MASTER_THESIS_CODE_ABSTRACTEXPR_H
