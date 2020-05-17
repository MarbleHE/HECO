#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERALFLOAT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERALFLOAT_H_

#include <string>
#include <ostream>
#include <map>
#include <unordered_map>
#include "AbstractLiteral.h"

template<typename T>
class Matrix;
class AbstractMatrix;

class LiteralFloat : public AbstractLiteral {
 public:
  explicit LiteralFloat(AbstractMatrix *am);

  explicit LiteralFloat(Matrix<AbstractExpr *> *am);

  explicit LiteralFloat(float value);

  LiteralFloat();

  explicit LiteralFloat(Matrix<float> *inputMatrix);

  ~LiteralFloat() override;

  LiteralFloat *clone(bool keepOriginalUniqueNodeId) const override;

  [[nodiscard]] float getValue() const;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  bool operator==(const LiteralFloat &rhs) const;

  bool operator!=(const LiteralFloat &rhs) const;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;

  void setValue(float val);

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  bool supportsDatatype(Datatype &datatype) override;

  void print(std::ostream &str) const override;

  bool isEqual(AbstractExpr *other) override;

  bool isNull() override;

  [[nodiscard]] AbstractMatrix *getMatrix() const override;

  void setMatrix(AbstractMatrix *newValue) override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_LITERALFLOAT_H_
