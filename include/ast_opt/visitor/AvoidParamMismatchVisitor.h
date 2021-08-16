
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

class SpecialInsertModSwitchVisitor : public ScopedVisitor {

  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;

 public:
  explicit SpecialAvoidParamMismatchVisitor( std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap)

  /// Visits an AST and based on the coefficient modulus map identifies binary expressions
  /// where a modswitch op needs to be inserted to avoid parameter mismatch
  /// pointer to resulting binary expressions will be stored in the vector modSwitchNodes
  /// \param node
  void visit(BinaryExpression &node);

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_AVOIDPARAMMISMATCHVISITOR_H_