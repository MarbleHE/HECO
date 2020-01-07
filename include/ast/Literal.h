#ifndef MASTER_THESIS_CODE_LITERAL_H
#define MASTER_THESIS_CODE_LITERAL_H

#include <variant>
#include <ostream>
#include "AbstractExpr.h"
#include <vector>

class Literal : public AbstractExpr {
 protected:
  ~Literal();
  virtual void print(std::ostream &str) const = 0;

 public:
  friend std::ostream &operator<<(std::ostream &os, const Literal &literal);

  bool operator==(const Literal &rhs) const;

  bool operator!=(const Literal &rhs) const;
};

inline std::ostream &operator<<(std::ostream &os, const std::vector<Literal*> &v) {
  os << "[";
  for (auto lit : v) { os << " " << *lit; }
  os << " ]" << std::endl;
  return os;
}

#endif //MASTER_THESIS_CODE_LITERAL_H
