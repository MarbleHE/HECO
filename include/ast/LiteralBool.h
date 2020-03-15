#ifndef AST_OPTIMIZER_INCLUDE_AST_LITERALBOOL_H_
#define AST_OPTIMIZER_INCLUDE_AST_LITERALBOOL_H_

#include <string>
#include <map>
#include <unordered_map>
#include "AbstractLiteral.h"

template<typename T>
class Matrix;
class AbstractMatrix;

class LiteralBool : public AbstractLiteral {
 public:
  explicit LiteralBool(AbstractMatrix *am);

  explicit LiteralBool(Matrix<AbstractExpr *> *am);

  explicit LiteralBool(bool value);

  explicit LiteralBool(Matrix<bool> *inputMatrix);

  ~LiteralBool() override;

  LiteralBool *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  bool getValue() const; /* NOLINT intentionally allow discarding value */

  [[nodiscard]] std::string getNodeType() const override;

  bool operator==(const LiteralBool &rhs) const;

  bool operator!=(const LiteralBool &rhs) const;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;

  void setValue(bool newValue);

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsDatatype(Datatype &datatype) override;

  void print(std::ostream &str) const override;

  bool supportsCircuitMode() override;

  bool isEqual(AbstractExpr *other) override;

  bool isNull() override;

  [[nodiscard]] AbstractMatrix *getMatrix() const override;

  void setMatrix(AbstractMatrix *newValue) override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_LITERALBOOL_H_
