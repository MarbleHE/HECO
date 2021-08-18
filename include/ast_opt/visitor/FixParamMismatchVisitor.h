
#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_FIXPARAMMISMATCHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_FIXPARAMMISMATCHVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <seal/seal.h>

#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/Visitor.h"


/// Forward declaration of the class that will actually implement the Visitor's logic
class SpecialFixParamMismatchVisitor;


/// uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialFixParamMismatchVisitor> FixParamMismatchVisitor;

class SpecialFixParamMismatchVisitor : public ScopedVisitor {

  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  std::vector<BinaryExpression *> modSwitchNodes;

 public:
  explicit SpecialFixParamMismatchVisitor( std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap,
                                             std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars);


  void visit(BinaryExpression &elem);

  void visit(Variable &elem);

  void visit(Assignment &elem);


  /// getter for coeffmodulus map
  /// \return coeffmodulusmap
  std::unordered_map<std::string, std::vector<seal::Modulus>> getCoeffModulusMap();

  /// getter for coeffmodulus map (key: variable identifier)
  /// \return coeffmodulusmap_vars
  std::unordered_map<std::string, std::vector<seal::Modulus>> getCoeffModulusMapVars();


};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_AVOIDPARAMMISMATCHVISITOR_H_
