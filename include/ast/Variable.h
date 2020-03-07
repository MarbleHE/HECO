#ifndef AST_OPTIMIZER_INCLUDE_AST_VARIABLE_H_
#define AST_OPTIMIZER_INCLUDE_AST_VARIABLE_H_

#include <map>
#include <string>
#include <vector>
#include "AbstractExpr.h"
#include "Matrix.h"

class Variable : public AbstractExpr {
 private:
  Matrix<std::string> *matrix;

 public:
  explicit Variable(Matrix<std::string> *inputMatrix);

  explicit Variable(std::string identifier);

  [[nodiscard]] json toJson() const override;

  ~Variable() override;

  Variable *clone(bool keepOriginalUniqueNodeId) override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  [[nodiscard]] std::string getIdentifier() const;

  bool operator==(const Variable &rhs) const;

  bool operator!=(const Variable &rhs) const;

  bool contains(Variable *var) override;

  bool isEqual(AbstractExpr *other) override;

  std::vector<std::string> getVariableIdentifiers() override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  [[nodiscard]] Matrix<std::string> *getMatrix() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_VARIABLE_H_
