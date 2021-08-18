#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/FixParamMismatchVisitor.h"
#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialFixParamMismatchVisitor::SpecialFixParamMismatchVisitor(std::unordered_map<std::string,
                                                                                  std::vector<seal::Modulus>> coeffmodulusmap,
                                                               std::unordered_map<std::string,
                                                                                  std::vector<seal::Modulus>> coeffmodulusmap_vars)
    : coeffmodulusmap(std::move(coeffmodulusmap)), coeffmodulusmap_vars(std::move(coeffmodulusmap_vars)) {}

void SpecialFixParamMismatchVisitor::visit(BinaryExpression &elem) {
  // recurse into left and right
  if (elem.hasLeft()) { elem.getLeft().accept(*this); }
  if (elem.hasRight()) { elem.getRight().accept(*this); }
  // potentially insert modswitches and update coeffmap

  int leftIndex;
  int rightIndex;

  // get coeff indices
  if (elem.hasLeft()) {
    leftIndex = coeffmodulusmap[elem.getLeft().getUniqueNodeId()].size();
  }
  if (elem.hasRight()) {
    rightIndex = coeffmodulusmap[elem.getRight().getUniqueNodeId()].size();
  }
  // if the indices are different, we need to insert modswitch(es)
  int diff = leftIndex - rightIndex;
  if (diff==0) {
    return;
  }
  // left child must be modswitched
  if (diff > 0) {
    // update coeffmodulus maps for binary expression and also the left child
    for (int i = 0; i < abs(diff); i++) {
      coeffmodulusmap[elem.getUniqueNodeId()].pop_back();
      coeffmodulusmap[elem.getLeft().getUniqueNodeId()].pop_back();
    }
    // insert modswitchs after left child
    auto leftNumModSw = std::make_unique<LiteralInt>(abs(diff));
    auto l = elem.takeLeft();
    std::vector<std::unique_ptr<AbstractExpression>> vLeft;
    vLeft.emplace_back(std::move(l));
    vLeft.emplace_back(std::move(leftNumModSw));
    auto cLeft = std::make_unique<Call>("modswitch", std::move(vLeft));
    cLeft->setParent(elem);
    elem.setLeft(std::move(cLeft));
  }
  // right child must me modswitched
  if (diff < 0) {
    // update coeffmodulus maps for binary expression and also the right child
    for (int i = 0; i < abs(diff); i++) {
      coeffmodulusmap[elem.getUniqueNodeId()].pop_back();
      coeffmodulusmap[elem.getRight().getUniqueNodeId()].pop_back();
    }
    // insert modswitchs after right child
    auto rightNumModSw = std::make_unique<LiteralInt>(abs(diff));
    auto r = elem.takeRight();
    std::vector<std::unique_ptr<AbstractExpression>> vRight;
    vRight.emplace_back(std::move(r));
    vRight.emplace_back(std::move(rightNumModSw));
    auto cRight = std::make_unique<Call>("modswitch", std::move(vRight));
    cRight->setParent(elem);
    elem.setRight(std::move(cRight));
  }
  return;
}

void SpecialFixParamMismatchVisitor::visit(Variable &elem) {
  coeffmodulusmap[elem.getUniqueNodeId()] = coeffmodulusmap_vars[elem.getIdentifier()];
}

void SpecialFixParamMismatchVisitor::visit(Assignment &elem) {
  AbstractTarget& at = elem.getTarget();
  Variable&  v = dynamic_cast<Variable&>(at);
  std::string ident = v.getIdentifier();
  coeffmodulusmap[elem.getUniqueNodeId()] = coeffmodulusmap_vars[ident];
}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialFixParamMismatchVisitor::getCoeffModulusMap() {
  return coeffmodulusmap;
}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialFixParamMismatchVisitor::getCoeffModulusMapVars() {
  return coeffmodulusmap_vars;
};

