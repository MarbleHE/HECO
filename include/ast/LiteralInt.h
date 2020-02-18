#ifndef AST_OPTIMIZER_INCLUDE_LITERALINT_H
#define AST_OPTIMIZER_INCLUDE_LITERALINT_H

#include "AbstractLiteral.h"
#include <string>
#include <ostream>
#include <map>

class LiteralInt : public AbstractLiteral {
 private:
  int value;
 public:
  explicit LiteralInt(int value);

  LiteralInt *clone(bool keepOriginalUniqueNodeId) override;

  ~LiteralInt() override;

  [[nodiscard]] int getValue() const;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  LiteralInt operator+(LiteralInt const &lint);

  friend std::ostream &operator<<(std::ostream &os, const LiteralInt &an_int);

  bool operator==(const LiteralInt &rhs) const;

  bool operator!=(const LiteralInt &rhs) const;

  bool supportsDatatype(Datatype &datatype) override;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;

  void setValue(int newValue);

  [[nodiscard]] std::string toString() const override;

  void print(std::ostream &str) const override;

  bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALINT_H
