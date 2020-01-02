#ifndef MASTER_THESIS_CODE_LITERALSTRING_H
#define MASTER_THESIS_CODE_LITERALSTRING_H

#include <string>
#include "Literal.h"

class LiteralString : public Literal, public Node {
 private:
  std::string value;
 public:
  explicit LiteralString(std::string value);

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] const std::string &getValue() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;
};

#endif //MASTER_THESIS_CODE_LITERALSTRING_H
