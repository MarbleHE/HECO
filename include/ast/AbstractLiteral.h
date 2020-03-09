#ifndef AST_OPTIMIZER_INCLUDE_AST_ABSTRACTLITERAL_H_
#define AST_OPTIMIZER_INCLUDE_AST_ABSTRACTLITERAL_H_

#include <variant>
#include <ostream>
#include <map>
#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "Datatype.h"
#include "Matrix.h"

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

  virtual bool isNull() = 0;

  virtual AbstractMatrix *getMatrix() const = 0;
};

inline std::ostream &operator<<(std::ostream &os, const std::vector<AbstractLiteral *> &v) {
  os << "[";
  for (auto lit : v) { os << " " << *lit; }
  os << " ]" << std::endl;
  return os;
}

#endif //AST_OPTIMIZER_INCLUDE_AST_ABSTRACTLITERAL_H_
