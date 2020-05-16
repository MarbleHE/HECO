#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTLITERAL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTLITERAL_H_

#include <variant>
#include <ostream>
#include <unordered_map>
#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "Datatype.h"

class RandLiteralGen;
class AbstractMatrix;

class AbstractLiteral : public AbstractExpr {
 protected:
  /// Stores the values of this Literal subtype. For example, for LiteralInt It can be a Matrix<int> but can also be a
  /// Matrix<AbstractExpr*> in the case that this matrix contains unevaluated expression, e.g., MatrixElementRef.
  AbstractMatrix *matrix;

  explicit AbstractLiteral(AbstractMatrix *matrix);

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

  [[nodiscard]] virtual AbstractMatrix *getMatrix() const = 0;

  virtual void setMatrix(AbstractMatrix *newValue) = 0;

  /// Creates a new and valueless literal matching the given datatype. For example, Types::INT would create a new
  /// LiteralInt and Types::BOOL a new LiteralBool.
  /// \param dt A pointer to a Datatype object.
  /// \return The created AbstractLiteral subtype instance.
  static AbstractLiteral *createLiteralBasedOnDatatype(Datatype *dt);

  std::vector<std::string> getVariableIdentifiers() override;
  std::vector<Variable *> getVariables() override;

};

inline std::ostream &operator<<(std::ostream &os, const std::vector<AbstractLiteral *> &v) {
  os << "[";
  for (auto lit : v) { os << " " << *lit; }
  os << " ]" << std::endl;
  return os;
}

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTLITERAL_H_
