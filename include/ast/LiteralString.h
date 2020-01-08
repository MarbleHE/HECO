#ifndef MASTER_THESIS_CODE_LITERALSTRING_H
#define MASTER_THESIS_CODE_LITERALSTRING_H

#include <string>
#include "Literal.h"

class LiteralString : public Literal {
 private:
  std::string value;

 protected:
  void print(std::ostream &str) const override;

 public:
  explicit LiteralString(std::string value);

  ~LiteralString() override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] const std::string &getValue() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  Literal* evaluate(Ast &ast) override;

  bool operator==(const LiteralString &rhs) const;

  bool operator!=(const LiteralString &rhs) const;

  void storeParameterValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) override;
};

#endif //MASTER_THESIS_CODE_LITERALSTRING_H
