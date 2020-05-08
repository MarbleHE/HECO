#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_

#include "Visitor.h"
#include <map>
#include "ast_opt/mockup_classes/Ciphertext.h"
#include "ast_opt/ast/Matrix.h"

class EvaluationVisitor;
class Ciphertext;
class AbstractMatrix;

/*
 * A helper struct to represent the return value of determineRequiredRotations that determines the required
 * rotations for ciphertexts.
 */
struct RotationData {
  int targetIndex;
  Ciphertext baseCiphertext;
  int requiredNumRotations;

  RotationData(int targetIndex, const Ciphertext &baseCiphertext, int requiredNumRotations)
      : targetIndex(targetIndex), baseCiphertext(baseCiphertext), requiredNumRotations(requiredNumRotations) {}
};

class RuntimeVisitor : public Visitor {
 private:
  // an instance of the EvaluationVisitor to be used to partially evaluate certain parts/expressions of the AST
  EvaluationVisitor *ev;

  // an enum that defines valid variable access types
  enum MatrixAccessMode { READ = 0, WRITE = 1 };

  // the access mode that is used for new entries in the variableAccessMap
  MatrixAccessMode currentMatrixAccessMode = READ;

  // tracks accessed matrix indices
  std::map<std::string,
           std::map<std::pair<int, int>, MatrixAccessMode>> matrixAccessMap;

  // stores for each variable (string) all created ciphertexts including rotated variants with their offset (int)
  std::map<std::string, std::map<int, Ciphertext>> varValues;

 public:
  explicit RuntimeVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues);

  void visit(For &elem) override;

  void visit(Ast &elem) override;

  void visit(MatrixElementRef &elem) override;

  void visit(MatrixAssignm &elem) override;

  void registerMatrixAccess(const std::string &variableIdentifier, int rowIndex, int columnIndex);

  void visit(VarDecl &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(Datatype &elem) override;

  static void checkIndexValidity(std::pair<int, int> rowColumnIndex);

  static std::vector<RotationData> determineRequiredRotations(std::map<int, Ciphertext> existingRotations,
                                                       const std::set<int>& reqIndices,
                                                       int targetSlot = -1);

  void executeRotations(const std::string &varIdentifier, const std::vector<RotationData> &reqRotations);
  void visit(OperatorExpr &elem) override;
};

template<typename T>
std::vector<double> transformToDoubleVector(AbstractMatrix *mx) {
  auto matrix = dynamic_cast<Matrix<T> *>(mx);
  std::vector<T> rawVector;
  if (matrix->getDimensions().equals(1, -1)) {
    rawVector = matrix->getNthRowVector(0);
  } else if (matrix->getDimensions().equals(-1, 1)) {
    rawVector = matrix->getNthColumnVector(0);
  } else {
    throw std::runtime_error("Given AbstractMatrix is neither a row (1,x) nor a column (x,1) vector!");
  }
  return std::vector<double>(rawVector.begin(), rawVector.end());
}

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_

