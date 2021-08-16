#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/AvoidParamMismatchVisitor.h"
#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialAvoidParamMismatchVisitor::SpecialAvoidParamMismatchVisitor(std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap)
    : coeffmodulusmap(std::move(coeffmodulusmap)) {}

void SpecialAvoidParamMismatchVisitor::visit(BinaryExpression &elem) {

// if binary expression's left child has children and has not been visited yet recurse into it
if ((elem.getLeft().begin() != elem.getLeft().end()) && (!isVisited[elem.getLeft().getUniqueNodeId()] || isVisited.count(elem.getLeft().getUniqueNodeId()) == 0)) {
  elem.getLeft().accept(*this);
}
// same for right
if ((elem.getRight().begin() != elem.getRight().end()) && (!isVisited[elem.getRight().getUniqueNodeId()]  || isVisited.count(elem.getRight().getUniqueNodeId()) == 0)) {
    elem.getLeft().accept(*this);
}
// base case
else {
    // set to 'visited'
    isVisited[elem.getUniqueNodeId()] = true;
    // check if modswitches need to be inserted and how many
    int diff = coeffmodulusmap[elem.getLeft().getUniqueNodeId()].size() - coeffmodulusmap[elem.getRight().getUniqueNodeId()].size();
    // if not, return
    if (diff == 0) {return;}
    else {
      modSwitchNodes.push_back(&elem);
      return;
    }

  }


};

