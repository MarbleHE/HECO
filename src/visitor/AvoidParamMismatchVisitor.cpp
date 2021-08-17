#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/AvoidParamMismatchVisitor.h"
#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialAvoidParamMismatchVisitor::SpecialAvoidParamMismatchVisitor(std::unordered_map<std::string,
                                                                   std::vector<seal::Modulus>> coeffmodulusmap)
    : coeffmodulusmap(std::move(coeffmodulusmap)){}

void SpecialAvoidParamMismatchVisitor::visit(BinaryExpression &elem) {
// if binary expression's left child has children and has not been visited yet recurse into it
  if (elem.hasLeft() && (elem.getLeft().begin()!=elem.getLeft().end())
      && (!isVisited[elem.getLeft().getUniqueNodeId()] || isVisited.count(elem.getLeft().getUniqueNodeId())==0)) {
    elem.getLeft().accept(*this);
  }
// same for right
  if (elem.hasRight() && (elem.getRight().begin()!=elem.getRight().end())
      && (!isVisited[elem.getRight().getUniqueNodeId()] || isVisited.count(elem.getRight().getUniqueNodeId())==0)) {
    elem.getLeft().accept(*this);
  }
// base case
  else {

    std::cout << "Visiting: " << elem.toString(false)<< std::endl;
    std::cout << "Left Child: " << elem.getLeft().toString(false) << std::endl;
    std::cout << "Right Child: " << elem.getRight().toString(false) << std::endl;
    std::cout << "Left cm: " << coeffmodulusmap[elem.getLeft().getUniqueNodeId()].size() << std::endl;
    std::cout << "Right cm: " << coeffmodulusmap[elem.getRight().getUniqueNodeId()].size() << std::endl;

    // set to 'visited'
    isVisited[elem.getUniqueNodeId()] = true;
    // check if modswitches need to be inserted and how many
    int diff = coeffmodulusmap[elem.getLeft().getUniqueNodeId()].size()
        - coeffmodulusmap[elem.getRight().getUniqueNodeId()].size();
    // if not, return
    if (diff==0) { return; }
    else {
      modSwitchNodes.push_back(&elem);
      return;
    }
  }
}

std::vector<BinaryExpression *> SpecialAvoidParamMismatchVisitor::getModSwitchNodes() {
  return modSwitchNodes;
};

