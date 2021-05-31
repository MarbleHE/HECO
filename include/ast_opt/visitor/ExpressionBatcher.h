#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_EXPRESSIONBATCHER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_EXPRESSIONBATCHER_H_
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/utilities/BatchingConstraint.h"
#include "ast_opt/utilities/Datatype.h"
#include "ast_opt/utilities/ComplexValue.h"
#include "ast_opt/utilities/VariableMap.h"
#include "ast_opt/utilities/ScopedVisitor.h"

class TreeNode;

// Forward Declaration for typedef below (must be above documentation, to ensure documentation is associated with the right type)
class SpecialExpressionBatcher;

typedef Visitor<SpecialExpressionBatcher> ExpressionBatcher;

class SpecialExpressionBatcher : public ScopedVisitor {
 private:
  typedef std::unordered_map<ScopedIdentifier, Datatype> TypeMap;
  /// Records the types of variables
  TypeMap types;

  typedef std::vector<ComplexValue> Values;
  /// Records pre-computed expression (as their execution plan)
  Values precomputedValues;

  /// Associates pre-computed values to variables
  VariableMap<ComplexValue&> variableValues;

  typedef std::unordered_map<ScopedIdentifier, BatchingConstraint> ConstraintMap;
  /// Records existing constraints on slot-encodings
  ConstraintMap constraints;



  /// Ugly hack to signal if an AbstractStatement just visited needs to be deleted.
  bool delete_flag = false;

 public:

  std::unique_ptr<TreeNode> batchExpression(AbstractExpression &expr, BatchingConstraint);

  std::unique_ptr<AbstractNode> computationTreeToAst(std::unique_ptr<TreeNode>&& computationTree);

  void visit(AbstractStatement &elem);

  /// Creates a Vectorizer without any pre-existing information
  SpecialExpressionBatcher() = default;

};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_EXPRESSIONBATCHER_H_
