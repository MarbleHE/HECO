#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"

#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialIdentifyNoisySubtreeVisitor::SpecialIdentifyNoisySubtreeVisitor(std::ostream &os, std::unordered_map<std::string, int> noise_map,
                                                   std::unordered_map<std::string, double> rel_noise_map, int encNoiseBudget)
    : os(os), noise_map(std::move(noise_map)), rel_noise_map(std::move(rel_noise_map)), encNoiseBudget(std::move(encNoiseBudget)) {}

void SpecialIdentifyNoisySubtreeVisitor::visit(BinaryExpression &elem) {

  if (elem.countChildren() > 1) {
    int leftNoiseBudget = noise_map.find(elem.getLeft().getUniqueNodeId())->second;
    int rightNoiseBudget = noise_map.find(elem.getRight().getUniqueNodeId())->second;

    if ((leftNoiseBudget==rightNoiseBudget) && (rightNoiseBudget==this->encNoiseBudget)) {
      std::string curNodeString = elem.toString(false);
      std::cout << "End";
      os << "CHECK NODE: " << curNodeString;
      os << " " << elem.getUniqueNodeId();
      return;
    } else if (leftNoiseBudget < rightNoiseBudget) {
      std::cout << "Going left: " << leftNoiseBudget << std::endl;
      elem.getLeft().accept(*this);
    } else if (leftNoiseBudget > rightNoiseBudget) {
      std::cout << "Going right: " << rightNoiseBudget << std::endl;
      elem.getRight().accept(*this);
    } else {
      elem.getLeft().accept(*this);
      elem.getRight().accept(*this);
    }
  }
}

void SpecialIdentifyNoisySubtreeVisitor::visit(BinaryExpression &elem, BinaryExpression &tail) {
  if (elem.countChildren() > 1) {
    int leftNoiseBudget = noise_map.find(elem.getLeft().getUniqueNodeId())->second;
    int rightNoiseBudget = noise_map.find(elem.getRight().getUniqueNodeId())->second;
    int currNoiseBudget = noise_map.find(elem.getUniqueNodeId())->second;

    if ((leftNoiseBudget==rightNoiseBudget) && (rightNoiseBudget==this->encNoiseBudget)) {
      std::string curNodeString = elem.toString(false);
      std::string tailNodeString = tail.toString(false);
      std::cout << "End";
      os << "CHECK NODE: " << curNodeString << "" ;
      os << " " << elem.getUniqueNodeId();
      return;
    } else if (leftNoiseBudget < rightNoiseBudget) {
      if (leftNoiseBudget == currNoiseBudget) {
        tail = elem; // check if this assignment works ie sets the tail to the current node
      }
      std::cout << "Going left: " << leftNoiseBudget << std::endl;
      elem.getLeft().accept(*this);
    } else if (leftNoiseBudget > rightNoiseBudget) {
      if (rightNoiseBudget == currNoiseBudget) {
        tail = elem; // check if this assignment works ie sets the tail to the current node
      }
      std::cout << "Going right: " << rightNoiseBudget << std::endl;
      elem.getRight().accept(*this);
    } else {
      if (leftNoiseBudget == currNoiseBudget) {
        tail = elem; // check if this assignment works ie sets the tail to the current node
      }
      elem.getLeft().accept(*this);
      if (rightNoiseBudget == currNoiseBudget) {
        tail = elem; // check if this assignment works ie sets the tail to the current node
      }
      elem.getRight().accept(*this);
    }
  }
}

