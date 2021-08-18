#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/AvoidParamMismatchVisitor.h"
#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialAvoidParamMismatchVisitor::SpecialAvoidParamMismatchVisitor(std::unordered_map<std::string,
                                                                   std::vector<seal::Modulus>> coeffmodulusmap,
                                                                   std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars)
    : coeffmodulusmap(std::move(coeffmodulusmap)),  coeffmodulusmap_vars(std::move(coeffmodulusmap_vars)){}

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
    // set to 'visited'
    isVisited[elem.getUniqueNodeId()] = true;
    // check if modswitches need to be inserted and how many
    // first we check if the operands left and right are binary expressions or variables to avoid bad casting and look up in the corresponding map
    int leftindex;
    int rightindex;
    if (dynamic_cast<Variable *>(&elem.getLeft())) {
      leftindex = coeffmodulusmap_vars[dynamic_cast<Variable &>(elem.getLeft()).getIdentifier()].size();
    } else if (!dynamic_cast<Variable *>(&elem.getLeft())) {
      leftindex = coeffmodulusmap[elem.getLeft().getUniqueNodeId()].size();
    }
    if (dynamic_cast<Variable *>(&elem.getRight())) {
      rightindex = coeffmodulusmap_vars[dynamic_cast<Variable &>(elem.getRight()).getIdentifier()].size();
    } else if (!dynamic_cast<Variable *>(&elem.getRight())) {
      rightindex =  coeffmodulusmap[elem.getRight().getUniqueNodeId()].size();
    }
      int diff = leftindex - rightindex;
    // if not, return
    if (diff==0) { return; }
    else {
      modSwitchNodes.push_back(&elem);
      return;
    }
  }
}

std::unique_ptr<AbstractNode> SpecialAvoidParamMismatchVisitor::insertModSwitchInAst(std::unique_ptr<AbstractNode> *ast,  BinaryExpression *binaryExpression) {
  // if no binary expression specified return original ast
  if (binaryExpression == nullptr) {return std::move(*ast);}

  // prepare argument for 'Call' node (modswitch)
  // we need to know how many modswitches to insert (will be second arg to ModSwitch call)
  int leftIndex;
  int rightIndex;

  if (dynamic_cast<Variable *>(&binaryExpression->getLeft())) {
    leftIndex = coeffmodulusmap_vars[dynamic_cast<Variable &>(binaryExpression->getLeft()).getIdentifier()].size() - 1;
  }
  else if (!dynamic_cast<Variable *>(&binaryExpression->getLeft())) {
    leftIndex = coeffmodulusmap[binaryExpression->getLeft().getUniqueNodeId()].size() - 1;
  }
  if  (dynamic_cast<Variable *>(&binaryExpression->getRight())) {
    rightIndex = coeffmodulusmap_vars[dynamic_cast<Variable &>(binaryExpression->getRight()).getIdentifier()].size() - 1;
  }
  else if (!dynamic_cast<Variable *>(&binaryExpression->getRight())) {
    rightIndex = coeffmodulusmap[binaryExpression->getRight().getUniqueNodeId()].size() -1;
  }

  int diff = leftIndex - rightIndex;

  // Note: only apply modswitches to var with more primes
  // if diff > 0, then the left side has more primes: need to switch left side only

  // get the result variable to update the coeffmodulusmap_vars for:
  VariableDeclaration& vd =  dynamic_cast<VariableDeclaration &>(binaryExpression->getParent());
  AbstractTarget& at = vd.getTarget();
  Variable&  v = dynamic_cast<Variable&>(at);
  std::string ident = v.getIdentifier();

  if (diff > 0) {
    // update coeffmodulus maps
    for (int i = 0; i < abs(diff); i++) {
      coeffmodulusmap[binaryExpression->getUniqueNodeId()].pop_back();
      if (dynamic_cast<Variable *>(&binaryExpression->getLeft())) {
        coeffmodulusmap_vars[dynamic_cast<Variable &>(binaryExpression->getLeft()).getIdentifier()].pop_back();
      }
      else {
        coeffmodulusmap[binaryExpression->getLeft().getUniqueNodeId()].pop_back();
      }
      coeffmodulusmap_vars[ident].pop_back();
    }
    // insert appropriate number of modswitches after left var
    auto leftNumModSw = std::make_unique<LiteralInt>(abs(diff));
    auto l = binaryExpression->takeLeft();
    std::vector<std::unique_ptr<AbstractExpression>> vLeft;
    vLeft.emplace_back(std::move(l));
    vLeft.emplace_back(std::move(leftNumModSw));
    auto cLeft = std::make_unique<Call>("modswitch", std::move(vLeft));
    cLeft->setParent(binaryExpression);
    binaryExpression->setLeft(std::move(cLeft));
  }
  else if(diff < 0) {
    // update coeffmodulus maps
    for (int i = 0; i < abs(diff); i++) {
      coeffmodulusmap[binaryExpression->getUniqueNodeId()].pop_back();
      if (dynamic_cast<Variable *>(&binaryExpression->getRight())) {
        coeffmodulusmap_vars[dynamic_cast<Variable &>(binaryExpression->getRight()).getIdentifier()].pop_back();
      }
      else {
        coeffmodulusmap[binaryExpression->getRight().getUniqueNodeId()].pop_back();
      }
      coeffmodulusmap_vars[ident].pop_back();
    }
    // insert appropriate number of modswitches after right var
    auto rightNumModSw = std::make_unique<LiteralInt>(abs(diff));
    auto r = binaryExpression->takeRight();
    std::vector<std::unique_ptr<AbstractExpression>> vRight;
    vRight.emplace_back(std::move(r));
    vRight.emplace_back(std::move(rightNumModSw));
    auto cRight = std::make_unique<Call>("modswitch", std::move(vRight));
    cRight->setParent(binaryExpression);
    binaryExpression->setRight(std::move(cRight));
  }
  return (std::move(*ast));
}

std::vector<BinaryExpression *> SpecialAvoidParamMismatchVisitor::getModSwitchNodes() {
  return modSwitchNodes;
}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialAvoidParamMismatchVisitor::getCoeffModulusMap() {
  return coeffmodulusmap;
}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialAvoidParamMismatchVisitor::getCoeffModulusMapVars() {
  return coeffmodulusmap_vars;
};

