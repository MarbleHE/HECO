#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/InsertModSwitchVisitor.h"


#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialInsertModSwitchVisitor::SpecialInsertModSwitchVisitor(std::ostream &os, std::unordered_map<std::string, int> noise_map,
                                                             std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap, int encNoiseBudget)
    : os(os), noise_map(std::move(noise_map)), coeffmodulusmap(std::move(coeffmodulusmap)), encNoiseBudget(std::move(encNoiseBudget)) {}

void SpecialInsertModSwitchVisitor::visit(BinaryExpression &elem) {

  // for the visited binary expression, calculate the noise budget, coeffmodulus chain length, and spent noise budgets of both operands,
  // as well as the difference of the chain length of both coeff modulus chains
  if (elem.countChildren() > 1) {
    int leftNoiseBudget = noise_map.find(elem.getLeft().getUniqueNodeId())->second;
    int rightNoiseBudget = noise_map.find(elem.getRight().getUniqueNodeId())->second;
    int leftIndex = coeffmodulusmap[elem.getLeft().getUniqueNodeId()].size() - 1;
    int rightIndex = coeffmodulusmap[elem.getRight().getUniqueNodeId()].size() - 1;
    int diff = leftIndex - rightIndex;
    int spentNoiseBudgetLeft = encNoiseBudget - leftNoiseBudget;
    int spentNoiseBudgetRight = encNoiseBudget - rightNoiseBudget;

    // if the sizes of the coeff modulus chains are different, we will want to insert |diff| + 1 many modswitches after one operand
    // and 1 modSwitch after the other

    // if diff > 0 then the right operand has fewer primes remaining in the coeffmodulus chain
    if (diff > 0) {
      int sum = 0; // this holds the sum of the bit length of the coeff_moduli in the chain to compare against spent noise budget
      for (int i = leftIndex; i >= (leftIndex - (abs(diff))); i--) {
        sum += coeffmodulusmap[elem.getLeft().getUniqueNodeId()][i].bit_count();
      }
      // if the spent noise budgets for the left and right operators are sufficiently large, we suggest a modswitch insertion procedure
      if ((sum < spentNoiseBudgetLeft) && spentNoiseBudgetRight > coeffmodulusmap[elem.getRight().getUniqueNodeId()][rightIndex].bit_count()) {
        modSwitchNodes.push_back(&elem);
      }
    }
    // if diff < 0 then the left operand has fewer primes remaining in the coeffmodulus chain
    else if (diff <0) {
      int sum = 0;
      for (int i = rightIndex; i >= (rightIndex - (abs(diff))); i--) {
        sum += coeffmodulusmap[elem.getRight().getUniqueNodeId()][i].bit_count();
      }
      // if the spent noise budgets for the left and right operators are sufficiently large, we suggest a modswitch insertion procedure
      if ((sum < spentNoiseBudgetRight) && spentNoiseBudgetLeft > coeffmodulusmap[elem.getLeft().getUniqueNodeId()][leftIndex].bit_count()) {
        modSwitchNodes.push_back(&elem);
      }
    }
    // if diff = 0, check if a single modswitch after both operands can be inserted
    else {
      if ((spentNoiseBudgetLeft > coeffmodulusmap[elem.getLeft().getUniqueNodeId()][leftIndex].bit_count()) &&
            (spentNoiseBudgetRight > coeffmodulusmap[elem.getRight().getUniqueNodeId()][rightIndex].bit_count())) {
        modSwitchNodes.push_back(&elem);
      }
    }
  }
  elem.getLeft().accept(*this);
  elem.getRight().accept(*this);

}

std::unique_ptr<AbstractNode> SpecialInsertModSwitchVisitor::insertModSwitchInAst(std::unique_ptr<AbstractNode> *ast,
                                                                                  BinaryExpression *binaryExpression,
                                                                                  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap) {

  // if no binary expression specified return original ast
  if (binaryExpression == nullptr) {return std::move(*ast);}


  // prepare argument for 'Call' node (modswitch)
  // we need to know how many modswitches to insert (will be second arg to ModSwitch call)
  int leftIndex = coeffmodulusmap[binaryExpression->getLeft().getUniqueNodeId()].size() - 1;
  int rightIndex = coeffmodulusmap[binaryExpression->getRight().getUniqueNodeId()].size() - 1;
  int diff = leftIndex - rightIndex;

  // take left and right child
  auto l = binaryExpression->takeLeft();
  auto r = binaryExpression->takeRight();

  std::vector<std::unique_ptr<AbstractExpression>> vLeft;
  std::vector<std::unique_ptr<AbstractExpression>> vRight;

  vLeft.emplace_back(std::move(l));
  vRight.emplace_back(std::move(r));

  // if diff > 0 then the right operand has fewer primes remaining in the coeffmodulus chain
  // etc...
  if (diff > 0) {
    auto leftNumModSw = std::make_unique<LiteralInt>(abs(diff) + 1);
    auto rightNumModSw = std::make_unique<LiteralInt>(1);
    vLeft.emplace_back(std::move(leftNumModSw));
    vRight.emplace_back(std::move(rightNumModSw));
  } else if (diff < 0) {
    auto rightNumModSw = std::make_unique<LiteralInt>(abs(diff) + 1);
    auto leftNumModSw = std::make_unique<LiteralInt>(1);
    vLeft.emplace_back(std::move(leftNumModSw));
    vRight.emplace_back(std::move(rightNumModSw));
  } else {
    auto rightNumModSw = std::make_unique<LiteralInt>(1);
    auto leftNumModSw = std::make_unique<LiteralInt>(1);
    vLeft.emplace_back(std::move(leftNumModSw));
    vRight.emplace_back(std::move(rightNumModSw));
  }

  auto cLeft = std::make_unique<Call>("modswitch", std::move(vLeft));
  auto cRight = std::make_unique<Call>("modswitch", std::move(vRight));

  // set parents to binaryexpr
  cLeft->setParent(binaryExpression);
  cRight->setParent(binaryExpression);

  // set children of binary expr to modswitchnodes
  binaryExpression->setLeft(std::move(cLeft));
  binaryExpression->setRight(std::move(cRight));

  return (std::move(*ast));
}

std::vector<BinaryExpression *> SpecialInsertModSwitchVisitor::getModSwitchNodes() const{
  return modSwitchNodes;
}

void SpecialInsertModSwitchVisitor::updateNoiseMap(AbstractNode& astProgram, RuntimeVisitor *srv) {
  // execute ast
  srv->executeAst(astProgram);
  this->noise_map = srv->getNoiseMap();
}

void SpecialInsertModSwitchVisitor::updateCoeffModulusMap(BinaryExpression *binaryExpression, int numSwitches) {
  //check that we can even drop that many primes
  if (numSwitches > coeffmodulusmap[binaryExpression->getUniqueNodeId()].size()) {
    std::runtime_error("Not possible to drop  primes: coeff modulus vector too short.");
  }
  // first, we update the coeffmodulus for the binary expression
  for (int i = 0; i < numSwitches; i++) {
    coeffmodulusmap[binaryExpression->getUniqueNodeId()].pop_back();
  }
  // now for all ancestors
  auto node = &binaryExpression->getParent();
  while (node !=nullptr) {
    for (int i = 0; i < numSwitches; i++) {
      coeffmodulusmap[node->getUniqueNodeId()].pop_back();
    }
    if (node->hasParent()) {
      node = &node->getParent();
    }
    else {return;}
  }
}

static std::unique_ptr<AbstractNode> removeModSwitchFromAst(std::unique_ptr<AbstractNode> *ast,
                                                            BinaryExpression *binaryExpression,
                                                            std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap){

  if (!binaryExpression || !binaryExpression->hasLeft() || !binaryExpression->hasRight()) {
    //throw exception
  }

  auto &l = binaryExpression->getLeft();
  auto &r = binaryExpression->getRight();

  if (auto clPointer = dynamic_cast<Call *>(&l)) {
    if (!clPointer->getArguments().empty() && clPointer->getIdentifier() == "modswitch") {
    //  auto x = clPointer->takeArgument(0); // returns unique pointer and ownership
    //  binaryExpression->setLeft(std::move(x));
    }
  }

  // do for right as well


  return std::move(*ast);
}

std::unique_ptr<AbstractNode> SpecialInsertModSwitchVisitor::rewriteAst(std::unique_ptr<AbstractNode> *ast, RuntimeVisitor srv,
                                         BinaryExpression *binaryExpression,
                                         std::unordered_map<std::string,
                                         std::vector<seal::Modulus>> coeffmodulusmap) {

  //1. identify sites eligible for modswitching
  auto binExprIns = this->getModSwitchNodes()[0];

  //2. insert modsw
  auto rewritten_ast = insertModSwitchInAst(ast, binExprIns, coeffmodulusmap);

  //3. recalc noise heurs
  updateNoiseMap(*rewritten_ast, &srv);

  //4. remove modswitch if necessary (i.e if root nodes noise budget is 0)
  //TODO


  //5. if modswitch was NOT removed, update coeffmodulusmap
  //5. return ast

  return nullptr;

}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialInsertModSwitchVisitor::getCoeffModulusMap() {
  return coeffmodulusmap;
}

std::unordered_map<std::string, int> SpecialInsertModSwitchVisitor::getNoiseMap() {
  return noise_map;
};


