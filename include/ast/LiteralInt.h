#ifndef MASTER_THESIS_CODE_LITERALINT_H
#define MASTER_THESIS_CODE_LITERALINT_H

#include "Literal.h"
#include <string>

class LiteralInt : public Literal, public Node {
 private:
  int value;

 public:
  [[nodiscard]] int getValue() const;

  explicit LiteralInt(int value);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;
};

#endif //MASTER_THESIS_CODE_LITERALINT_H
