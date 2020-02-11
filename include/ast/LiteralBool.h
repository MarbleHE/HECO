#ifndef MASTER_THESIS_CODE_LITERALBOOL_H
#define MASTER_THESIS_CODE_LITERALBOOL_H

#include "Literal.h"
#include <string>
#include <map>

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

  void addLiteralValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;
  void setValue(bool newValue);

  std::string toString() const override;
  bool supportsDatatype(Datatype &datatype) override;

 protected:
  void print(std::ostream &str) const override;
  bool supportsCircuitMode() override;
};

#endif //MASTER_THESIS_CODE_LITERALBOOL_H
