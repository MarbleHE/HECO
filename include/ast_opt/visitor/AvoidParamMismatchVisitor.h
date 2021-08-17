
#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_AVOIDPARAMMISMATCHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_AVOIDPARAMMISMATCHVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <seal/seal.h>

#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/Visitor.h"


/// Forward declaration of the class that will actually implement the Visitor's logic
class SpecialAvoidParamMismatchVisitor;


/// uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialAvoidParamMismatchVisitor> AvoidParamMismatchVisitor;

class SpecialAvoidParamMismatchVisitor : public ScopedVisitor {

  /// map unique_node_id --> bool that indicates if a node has already been visited
  std::unordered_map<std::string, bool> isVisited;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  std::vector<BinaryExpression *> modSwitchNodes;

 public:
  explicit SpecialAvoidParamMismatchVisitor( std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap,
                                             std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars);

  /// Visits an AST and based on the coefficient modulus map identifies binary expressions
  /// where a modswitch op needs to be inserted to avoid parameter mismatch. pushes them into the vector modSwitchNodes
  /// \param node
  void visit(BinaryExpression &elem);

  /// insert modswittch as needed to fix the potential parameter mismatch caused by the InsertModSwitchVisitor
  /// \param ast to rewrite
  /// \param binary expression as ientified from the AvoidParamMismatch visitor (visit(BinaryExpression &elem))
  /// \return a rewritten ast
  std::unique_ptr<AbstractNode> insertModSwitchInAst(std::unique_ptr<AbstractNode> *ast,  BinaryExpression *binaryExpression = nullptr);


  /// getter function
  /// \return modSwitch nodes: Binary expressions whose children need to be modswitched to ensure correctness of the circuit.
  std::vector<BinaryExpression *> getModSwitchNodes();

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_AVOIDPARAMMISMATCHVISITOR_H_
