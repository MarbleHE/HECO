#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_HASHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_HASHVISITOR_H_

#include <unordered_map>
#include <stack>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/ScopedVisitor.h"


/// Forward declaration of the class that will actually implement the HashVisitor's logic
class SpecialHashVisitor;

/// HashVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialHashVisitor> HashVisitor;

class SpecialHashVisitor : public ScopedVisitor {
 private:
  /// Reference to the hashmap we have to build
  std::unordered_map<std::string, std::string>& map;

 public:
  explicit SpecialHashVisitor(std::unordered_map<std::string, std::string>& map);

  void visit(AbstractNode&);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_HASHVISITOR_H_
