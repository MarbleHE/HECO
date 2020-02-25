#ifndef AST_OPTIMIZER_INCLUDE_LITERALSTRING_H
#define AST_OPTIMIZER_INCLUDE_LITERALSTRING_H

#include <string>
#include "AbstractLiteral.h"
#include <map>

class LiteralString : public AbstractLiteral {
 private:
  std::string value;

 protected:
  void print(std::ostream &str) const override;

 public:
  explicit LiteralString(std::string value);

  ~LiteralString() override;

  LiteralString *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] const std::string &getValue() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  bool operator==(const LiteralString &rhs) const;

  bool operator!=(const LiteralString &rhs) const;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setValue(const std::string &newValue);

  void setRandomValue(RandLiteralGen &rlg) override;

  [[nodiscard]] std::string toString() const override;

  bool supportsCircuitMode() override;

  bool supportsDatatype(Datatype &datatype) override;

  bool isEqual(AbstractExpr *other) override;
  bool isNull() override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALSTRING_H
