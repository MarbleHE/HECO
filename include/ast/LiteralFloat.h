#ifndef MASTER_THESIS_CODE_LITERALFLOAT_H
#define MASTER_THESIS_CODE_LITERALFLOAT_H

#include "Literal.h"
#include <string>
#include <ostream>
#include <map>

class LiteralFloat : public Literal {
 private:
  float value;

 protected:
  void print(std::ostream &str) const override;

 public:
  explicit LiteralFloat(float value);

  ~LiteralFloat();

  [[nodiscard]] float getValue() const;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  Literal* evaluate(Ast &ast) override;

  LiteralFloat operator+(LiteralFloat const &lfloat);

  friend std::ostream &operator<<(std::ostream &os, const LiteralFloat &an_float);

  bool operator==(const LiteralFloat &rhs) const;

  bool operator!=(const LiteralFloat &rhs) const;

  void addLiteralValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) override;

  void setRandomValue(RandLiteralGen &rlg) override;
  void setValue(float val);
};

#endif //MASTER_THESIS_CODE_LITERALFLOAT_H
