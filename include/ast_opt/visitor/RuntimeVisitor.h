#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_

#include "Visitor.h"
#include <map>

class EvaluationVisitor;
class RuntimeVisitor : public Visitor {
 private:
  EvaluationVisitor *ev;

  enum MatrixAccessMode { READ = 0, WRITE = 1 };
  MatrixAccessMode currentMatrixAccessMode = READ;

  std::map<std::string,
           std::map<std::pair<int, int>, MatrixAccessMode>> variableAccessMap;

 public:
  explicit RuntimeVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues);

  void visit(For &elem) override;

  void visit(Ast &elem) override;

  void visit(MatrixElementRef &elem) override;

  void visit(MatrixAssignm &elem) override;

  void registerMatrixAccess(std::string variableIdentifier, int rowIndex, int columnIndex);

  void visit(VarDecl &elem) override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_RUNTIMEVISITOR_H_

