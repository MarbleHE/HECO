#ifndef MASTER_THESIS_CODE_LITERALINT_H
#define MASTER_THESIS_CODE_LITERALINT_H

#include "Literal.h"
#include <string>
#include <ostream>

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

  void storeParameterValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) override;

 protected:
  void print(std::ostream &str) const override;
};

#endif //MASTER_THESIS_CODE_LITERALINT_H
