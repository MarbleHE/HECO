#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_

#define NDEBUG 1

#include "Visitor.h"
#include <map>
#include <utility>
#include <vector>
#include <string>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include "ast_opt/mockup_classes/Ciphertext.h"
#include "ast_opt/ast/Matrix.h"



class EvaluationVisitor;
class SecretTaintingVisitor;
class Ciphertext;
class AbstractMatrix;

/*
 * A helper struct to represent the return value of determineRequiredRotations that determines the required
 * rotations for ciphertexts.
 */
struct RotationData {
  int targetIndex;
  Ciphertext *baseCiphertext;
  int requiredNumRotations;

  RotationData(int targetIndex, Ciphertext *baseCiphertext, int requiredNumRotations)
      : targetIndex(targetIndex), baseCiphertext(baseCiphertext), requiredNumRotations(requiredNumRotations) {}
};

/*
 * TODO comment me!
 */
struct MatrixElementAccess {
  int rowIndex;
  int columnIndex;
  std::string variable;
  MatrixElementAccess(int rowIndex, int columnIndex, std::string variable)
      : rowIndex(rowIndex), columnIndex(columnIndex), variable(std::move(std::move(variable))) {}

  inline bool operator<(const MatrixElementAccess &rhs) const {
    return variable < rhs.variable
        || (variable==rhs.variable && rowIndex < rhs.rowIndex)
        || (variable==rhs.variable && rowIndex==rhs.rowIndex && columnIndex < rhs.columnIndex);
  }
};

struct VarValuesEntry {
  Ciphertext *ctxt;
  AbstractExpr *expr;
  VarValuesEntry() : ctxt(nullptr), expr(nullptr) {}

  VarValuesEntry &operator=(const VarValuesEntry &other) {
    ctxt = new Ciphertext(*other.ctxt);
    expr = other.expr->clone(false)->castTo<AbstractExpr>();
    return *this;
  }

  VarValuesEntry(Ciphertext *ctxt, AbstractExpr *expr) : ctxt(ctxt), expr(expr) {}
  explicit VarValuesEntry(Ciphertext *ctxt) : ctxt(ctxt), expr(nullptr) {}
};

typedef std::map<int, VarValuesEntry> VarValuesMap;

// an enum that defines valid variable access types
enum MatrixAccessMode { READ = 0, WRITE = 1 };

// TODO comment me!
enum TraversalPass { ANALYSIS, EVALUATION_PLAINTEXT, EVALUATION_CIPHERTEXT };

class RuntimeVisitor : public Visitor {
 public:
  /// Toggle to disable all batching optimizations in this function
  bool disableBatchingOpt = false;
 private:
  // an instance of the EvaluationVisitor to be used to evaluate expressions of the AST
  EvaluationVisitor *ev;

  // a list of tainted nodes of the current or last visited AST
  std::unordered_set<std::string> taintedNodesUniqueIds;

  std::vector<Ciphertext*> returnValues;

  // a list of unique node IDs for that the required ciphertext is created
  std::map<MatrixElementAccess, VarValuesMap::iterator> precomputedCiphertexts;

  std::unordered_set<std::string> precomputedStatements;

  TraversalPass visitingForEvaluation;

  // the access mode that is used for new entries in the variableAccessMap
  MatrixAccessMode currentMatrixAccessMode = READ;

  AbstractStatement *lastVisitedStatement;

  std::stack<Ciphertext *> intermedResult;

  // tracks accessed matrix indices
  std::map<std::string,
           std::map<std::pair<int, int>, MatrixAccessMode>> matrixAccessMap;

  // stores for each variable (string) all created ciphertexts including rotated variants with their offset (int)
  std::map<std::string, VarValuesMap> varValues;

  static void checkIndexValidity(std::pair<int, int> rowColumnIndex);

  static std::vector<RotationData> determineRequiredRotations(VarValuesMap existingRotations,
                                                              const std::unordered_set<int> &reqIndices,
                                                              int targetSlot = -1);

  std::vector<VarValuesMap::iterator> executeRotations(const std::string &varIdentifier,
                                                       const std::vector<RotationData> &reqRotations);

  void registerMatrixAccess(const std::string &variableIdentifier, int rowIndex, int columnIndex);

  void clearIntermediateResults(int numElementsBeforeVisitingChildren = -1);

 public:
  explicit RuntimeVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues);

  void visit(For &elem) override;

  void visit(Ast &elem) override;

  void visit(MatrixElementRef &elem) override;

  void visit(MatrixAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(Datatype &elem) override;

  void visit(OperatorExpr &elem) override;
  void visit(Return &elem) override;
  void visit(Variable &elem) override;
  const std::vector<Ciphertext *> &getReturnValues() const;
  void visit(VarAssignm &elem) override;
  void visit(ArithmeticExpr &elem) override;
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
    throw std::runtime_error("Given AbstractMatrix is neither a (1,x)-row nor a (x,1)-column vector!");
  }
  return std::vector<double>(rawVector.begin(), rawVector.end());
}

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_

