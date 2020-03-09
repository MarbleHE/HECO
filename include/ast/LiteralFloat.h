#ifndef AST_OPTIMIZER_INCLUDE_AST_LITERALFLOAT_H_
#define AST_OPTIMIZER_INCLUDE_AST_LITERALFLOAT_H_

#include "AbstractLiteral.h"
#include <string>
#include <ostream>
#include <map>
#include <unordered_map>
#include "Matrix.h"

class LiteralFloat : public AbstractLiteral {
 private:
  Matrix<float> *matrix;

 public:
  explicit LiteralFloat(float value);

  explicit LiteralFloat(Matrix<float> *inputMatrix);

  ~LiteralFloat() override;

  LiteralFloat *clone(bool keepOriginalUniqueNodeId) override;

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

  CMatrix *getMatrix() const override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_LITERALFLOAT_H_
