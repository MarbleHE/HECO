#ifndef MASTER_THESIS_CODE_ABSTRACTEXPR_H
#define MASTER_THESIS_CODE_ABSTRACTEXPR_H

#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "../visitor/Visitor.h"
#include "Ast.h"
#include "Node.h"

using json = nlohmann::json;

class AbstractExpr : public Node {
 protected:
  static LiteralInt* createParam(int i);

  static LiteralBool* createParam(bool b);

  static LiteralString* createParam(const char* str);

  static LiteralFloat* createParam(float f);

  static AbstractExpr* createParam(AbstractExpr* abstractExpr);

 public:
  virtual ~AbstractExpr() = default;

  [[nodiscard]] std::string toString() const override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  virtual BinaryExpr* contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree);

  virtual int countByTemplate(AbstractExpr* abstractExpr);

  virtual std::vector<std::string> getVariableIdentifiers();

  virtual bool contains(Variable* var);

  virtual bool isEqual(AbstractExpr* other);

  Literal* evaluate(Ast &ast) override;
};

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj);

#endif //MASTER_THESIS_CODE_ABSTRACTEXPR_H
