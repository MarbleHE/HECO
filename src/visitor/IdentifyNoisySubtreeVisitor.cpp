#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"

#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialIdentifyNoisySubtreeVisitor::SpecialIdentifyNoisySubtreeVisitor(std::ostream &os, std::unordered_map<std::string, int> noise_map,
                                                   std::unordered_map<std::string, double> rel_noise_map)
    : os(os), noise_map(std::move(noise_map)), rel_noise_map(std::move(rel_noise_map)) {}

void SpecialIdentifyNoisySubtreeVisitor::visit(BinaryExpression &elem) {
  if (elem.countChildren() > 1) {
    int leftNoiseBudget = noise_map.find(elem.getLeft().getUniqueNodeId())->second;
    int rightNoiseBudget =  noise_map.find(elem.getRight().getUniqueNodeId())->second;
    int encNoiseBudget = 128;

    if ((leftNoiseBudget == rightNoiseBudget) && (rightNoiseBudget == encNoiseBudget)) {
      std::string curNodeString = elem.toString(false);
      os << "CHECK NODE: " << curNodeString;
      os << " " << elem.getUniqueNodeId();
      return;
    }
    else if (leftNoiseBudget < rightNoiseBudget) {
      std::cout << "Going left: " << leftNoiseBudget << std::endl;
      elem.getLeft().accept(*this);
    }
    else if (leftNoiseBudget > rightNoiseBudget) {
      std::cout << "Going right: " << rightNoiseBudget << std::endl;
      elem.getRight().accept(*this);
    }
    else {
      elem.getLeft().accept(*this);
      elem.getRight().accept(*this);
    }
  }
}
