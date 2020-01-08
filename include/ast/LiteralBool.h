#ifndef MASTER_THESIS_CODE_LITERALBOOL_H
#define MASTER_THESIS_CODE_LITERALBOOL_H

#include "Literal.h"
#include <string>

class LiteralBool : public Literal {
 private:
  bool value;

 public:
  explicit LiteralBool(bool value);

  ~LiteralBool() override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] bool getValue() const;

  [[nodiscard]] std::string getTextValue() const;

  [[nodiscard]] std::string getNodeName() const override;

  Literal* evaluate(Ast &ast) override;

  bool operator==(const LiteralBool &rhs) const;

  bool operator!=(const LiteralBool &rhs) const;

  void storeParameterValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) override;

 protected:
  void print(std::ostream &str) const override;
};

#endif //MASTER_THESIS_CODE_LITERALBOOL_H
