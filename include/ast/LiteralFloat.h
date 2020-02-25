#ifndef AST_OPTIMIZER_INCLUDE_LITERALFLOAT_H
#define AST_OPTIMIZER_INCLUDE_LITERALFLOAT_H

#include "AbstractLiteral.h"
#include <string>
#include <ostream>
#include <map>

class LiteralFloat : public AbstractLiteral {
 private:
  float value;

 public:
  explicit LiteralFloat(float value);

  ~LiteralFloat() override;

  LiteralFloat *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] float getValue() const;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  LiteralFloat operator+(LiteralFloat const &lfloat);

  friend std::ostream &operator<<(std::ostream &os, const LiteralFloat &an_float);

  bool operator==(const LiteralFloat &rhs) const;

  bool operator!=(const LiteralFloat &rhs) const;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;

  void setValue(float val);

  [[nodiscard]] std::string toString() const override;

  bool supportsCircuitMode() override;

  bool supportsDatatype(Datatype &datatype) override;

  void print(std::ostream &str) const override;
  bool isEqual(AbstractExpr *other) override;
  bool isNull() override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALFLOAT_H
