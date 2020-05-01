#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERALSTRING_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERALSTRING_H_

#include <string>
#include <unordered_map>
#include "AbstractLiteral.h"

template<typename T>
class Matrix;
class AbstractMatrix;

class LiteralString : public AbstractLiteral {
 public:
  explicit LiteralString(AbstractMatrix *pMatrix);

  explicit LiteralString(Matrix<AbstractExpr *> *am);

  explicit LiteralString(Matrix<std::string> *inputMatrix);

  explicit LiteralString(std::string value);

  LiteralString();

  ~LiteralString() override;

  void print(std::ostream &str) const override;

  LiteralString *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] std::string getValue() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  bool operator==(const LiteralString &rhs) const;

  bool operator!=(const LiteralString &rhs) const;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setValue(const std::string &newValue);

  void setRandomValue(RandLiteralGen &rlg) override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  bool supportsDatatype(Datatype &datatype) override;

  bool isEqual(AbstractExpr *other) override;

  bool isNull() override;

  [[nodiscard]] AbstractMatrix *getMatrix() const override;

  void setMatrix(AbstractMatrix *newValue) override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERALSTRING_H_
