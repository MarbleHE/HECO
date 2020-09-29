#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/HashVisitor.h"

SpecialHashVisitor::SpecialHashVisitor(std::unordered_map<std::string, std::string> &map) : map(map) {}

void SpecialHashVisitor::visit(AbstractNode &elem) {

  //TODO: Implement actually meaningful hash based on values/structures rather than uniqueID
  std::string hash = elem.getUniqueNodeId();

  map.insert({elem.getUniqueNodeId(),hash});

  // visit children
  for(auto &c: elem) {
    c.accept(*this);
  }

}