#ifndef AST_OPTIMIZER_INCLUDE_LITERALINT_H
#define AST_OPTIMIZER_INCLUDE_LITERALINT_H

#include "Literal.h"
#include <string>
#include <ostream>
#include <map>

class LiteralInt : public Literal {
 private:
  int value;

 public:
  explicit LiteralInt(int value);

  ~LiteralInt() override;

  [[nodiscard]] int getValue() const;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  Literal* evaluate(Ast &ast) override;

  LiteralInt operator+(LiteralInt const &lint);

  friend std::ostream &operator<<(std::ostream &os, const LiteralInt &an_int);

  bool operator==(const LiteralInt &rhs) const;

  bool operator!=(const LiteralInt &rhs) const;

  bool supportsDatatype(Datatype &datatype) override;

  void addLiteralValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;

  void setValue(int newValue);

  [[nodiscard]] std::string toString() const override;

  void print(std::ostream &str) const override;

  bool supportsCircuitMode() override;
 private:
  Node* createClonedNode(bool keepOriginalUniqueNodeId) override;
};

#endif //AST_OPTIMIZER_INCLUDE_LITERALINT_H
