#ifndef MASTER_THESIS_CODE_LITERAL_H
#define MASTER_THESIS_CODE_LITERAL_H

#include <variant>
#include <ostream>
#include "AbstractExpr.h"

class Literal : public AbstractExpr {
 protected:
  ~Literal();
  virtual void print(std::ostream & str) const = 0;

 public:
  friend std::ostream &operator<<(std::ostream &os, const Literal &literal);
};

#endif //MASTER_THESIS_CODE_LITERAL_H
