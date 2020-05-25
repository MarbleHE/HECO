#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_GETMATRIXSIZE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_GETMATRIXSIZE_H_

#include "AbstractExpr.h"
#include <string>

/**
 * A node representing the retrieval of a matrix's dimension.
 *
 * If the given matrix is known at compile-time, the method returns the specified dimension of the matrix when
 * executing the EvaluationVisitor. For example, if the 2x3 matrix is defined as M = [3 2 1; 5 3 3], then
 * GetMatrixSize(M, 0) returns 2 and GetMatrixSize(M, 1) returns 3. Requesting any other (non-existing) higher
 * dimension returns 0.
 *
 * If the matrix is not known at compile-time, e.g., a function's input parameter then the downstream system
 * generating the FHE code (e.g., WOOL) must handle the method's implementation by replacing the function's call by its
 * return value (i.e., the requested matrix dimension).
 */
class GetMatrixSize : public AbstractExpr {
 public:
  GetMatrixSize(AbstractExpr *matrix, AbstractExpr *requestedDimension);

  ~GetMatrixSize() override;

  [[nodiscard]] json toJson() const override;

  bool isEqual(AbstractExpr *other) override;

  [[nodiscard]] std::string getNodeType() const override;

  int getMaxNumberChildren() override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override;

  bool supportsCircuitMode() override;

  [[nodiscard]] AbstractExpr *getMatrixOperand() const;

  [[nodiscard]] AbstractExpr *getDimensionParameter() const;
  std::vector<std::string> getVariableIdentifiers() override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_GETMATRIXSIZE_H_
