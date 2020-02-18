#ifndef AST_OPTIMIZER_INCLUDE_LITERAL_H
#define AST_OPTIMIZER_INCLUDE_LITERAL_H

#include <variant>
#include <ostream>
#include <map>
#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "Datatype.h"

class RandLiteralGen;

class AbstractLiteral : public AbstractExpr {
 protected:
  ~AbstractLiteral() override;

  virtual void print(std::ostream &str) const = 0;

 public:
  friend std::ostream &operator<<(std::ostream &os, const AbstractLiteral &literal);

  bool operator==(const AbstractLiteral &rhs) const;

  bool operator!=(const AbstractLiteral &rhs) const;

  virtual void addLiteralValue(std::string identifier,
                               std::unordered_map<std::string, AbstractLiteral *> &paramsMap) = 0;

  virtual void setRandomValue(RandLiteralGen &rlg) = 0;

  virtual bool supportsDatatype(Datatype &datatype) = 0;
};

inline std::ostream &operator<<(std::ostream &os, const std::vector<AbstractLiteral *> &v) {
  os << "[";
  for (auto lit : v) { os << " " << *lit; }
  os << " ]" << std::endl;
  return os;
}

#endif //AST_OPTIMIZER_INCLUDE_LITERAL_H
