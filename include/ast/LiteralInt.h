#ifndef AST_OPTIMIZER_INCLUDE_AST_LITERALINT_H_
#define AST_OPTIMIZER_INCLUDE_AST_LITERALINT_H_

#include <string>
#include <ostream>
#include <map>
#include "AbstractLiteral.h"
#include "Matrix.h"

class LiteralInt : public AbstractLiteral {
 private:
  Matrix<int> *matrix;

 public:
  explicit LiteralInt(Matrix<int> *inputMatrix);

  explicit LiteralInt(int value);

  LiteralInt *clone(bool keepOriginalUniqueNodeId) override;

  ~LiteralInt() override;

  [[nodiscard]] int getValue() const;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  bool operator==(const LiteralInt &rhs) const;

  bool operator!=(const LiteralInt &rhs) const;

  bool supportsDatatype(Datatype &datatype) override;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;

  void setValue(int newValue);

  [[nodiscard]] std::string toString(bool printChildren) const override;

  void print(std::ostream &str) const override;

  bool supportsCircuitMode() override;

  bool isEqual(AbstractExpr *other) override;

  bool isNull() override;

  CMatrix *getMatrix() const override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_LITERALINT_H_
