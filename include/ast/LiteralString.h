#ifndef AST_OPTIMIZER_INCLUDE_AST_LITERALSTRING_H_
#define AST_OPTIMIZER_INCLUDE_AST_LITERALSTRING_H_

#include <string>
#include <map>
#include "AbstractLiteral.h"
#include "Matrix.h"

class LiteralString : public AbstractLiteral {
 private:
  Matrix<std::string> *matrix;

 public:
  explicit LiteralString(Matrix<std::string> *inputMatrix);

  explicit LiteralString(std::string value);

  ~LiteralString() override;

  void print(std::ostream &str) const override;

  LiteralString *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] std::string getValue() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  bool operator==(const LiteralString &rhs) const;

  bool operator!=(const LiteralString &rhs) const;

  void addLiteralValue(std::string identifier, std::unordered_map<std::string, AbstractLiteral *> &paramsMap) override;

  void setValue(const std::string &newValue);

  void setRandomValue(RandLiteralGen &rlg) override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  bool supportsDatatype(Datatype &datatype) override;

  bool isEqual(AbstractExpr *other) override;

  bool isNull() override;

  [[nodiscard]] Matrix<std::string> *getMatrix() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_LITERALSTRING_H_
