#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/InsertModSwitchVisitor.h"
#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialInsertModSwitchVisitor::SpecialInsertModSwitchVisitor(std::ostream &os, std::unordered_map<std::string, int> noise_map,
                                                             std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap,
                                                             std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars,
                                                             int encNoiseBudget)
    : os(os), noise_map(std::move(noise_map)), coeffmodulusmap(std::move(coeffmodulusmap)), coeffmodulusmap_vars(std::move(coeffmodulusmap_vars)), encNoiseBudget(std::move(encNoiseBudget)) {}

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

  // update coeff modulus maps
  updateCoeffModulusMap(binaryExpression, abs(diff) + 1);

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

  std::cout << "Update " << binaryExpression->getParent().getParent().toString(false) << std::endl;
   //std::cout << "Update " << dynamic_cast<Variable &>(binaryExpression->getParent()).getIdentifier() << std::endl;
  // std::cout << "IDentifierLeft " << dynamic_cast<Variable *>(&binaryExpression->getLeft())->getIdentifier() << std::endl;
  //std::cout << "IDentifierRight " << dynamic_cast<Variable *>(&binaryExpression->getRight())->getIdentifier() << std::endl;

  if (numSwitches > coeffmodulusmap[binaryExpression->getUniqueNodeId()].size()) {
    std::runtime_error("Not possible to drop  primes: coeff modulus vector too short.");
  }
  // first, we update the coeffmodulus for the binary expression as well as for the modswitched variables
  for (int i = 0; i < numSwitches; i++) {
    coeffmodulusmap[binaryExpression->getUniqueNodeId()].pop_back();
    coeffmodulusmap_vars[dynamic_cast<Variable &>(binaryExpression->getLeft()).getIdentifier()].pop_back();
    coeffmodulusmap_vars[dynamic_cast<Variable &>(binaryExpression->getRight()).getIdentifier()].pop_back();
  }
}

std::unique_ptr<AbstractNode> SpecialInsertModSwitchVisitor::removeModSwitchFromAst(std::unique_ptr<AbstractNode> *ast,
                                                            BinaryExpression *binaryExpression,
                                                            std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap){

  if (!binaryExpression || !binaryExpression->hasLeft() || !binaryExpression->hasRight()) {
    throw std::runtime_error("Not able to remove modswitches: Either no binary expression or something else went wrong.");
  }

  auto &l = binaryExpression->getLeft();
  auto &r = binaryExpression->getRight();

  if (auto clPointer = dynamic_cast<Call *>(&l)) {
    if (!clPointer->getArguments().empty() && clPointer->getIdentifier() == "modswitch") {
      auto x = clPointer->takeArgument(0); // returns unique pointer and ownership
      binaryExpression->setLeft(std::move(x));
    }
  }

  // do for right as well
  if (auto crPointer = dynamic_cast<Call *>(&r)) {
    if (!crPointer->getArguments().empty() && crPointer->getIdentifier() == "modswitch") {
        auto x = crPointer->takeArgument(0); // returns unique pointer and ownership
        binaryExpression->setRight(std::move(x));
    }
  }
  return std::move(*ast);
}

std::unique_ptr<AbstractNode> SpecialInsertModSwitchVisitor::rewriteAst(std::unique_ptr<AbstractNode> *ast, RuntimeVisitor srv,
                                         std::unordered_map<std::string,
                                         std::vector<seal::Modulus>> coeffmodulusmap) {



  //1. identify sites eligible for modswitching
  auto binExprIns = this->getModSwitchNodes();

  std::unique_ptr<AbstractNode> rewritten_ast;

  // for each site try to insert a modswitch
  for (int i = 1; i < binExprIns.size(); i++) {
    std::cout << "inserting modswitch at binary expr: " << binExprIns[i]->toString(false)<< std::endl;
    //2. insert modsw
    rewritten_ast = insertModSwitchInAst(ast, binExprIns[i], coeffmodulusmap);

    std::cout << "...Done" <<std::endl;

    //3. recalc noise heurs

    std::cout << "Updating noise map... " << std::endl;

    updateNoiseMap(*rewritten_ast, &srv);

    std::cout << "...Done" <<std::endl;
    //4. remove modswitch if necessary (i.e if root nodes noise budget is 0)
    auto noiseMap = srv.getNoiseMap();

    std::cout << "noise at root of the new ast: " << noiseMap[rewritten_ast->getUniqueNodeId()] << std::endl;
    if (noiseMap[rewritten_ast->getUniqueNodeId()] == 0) {
      rewritten_ast = removeModSwitchFromAst(&rewritten_ast);
    } else {
      // how many modswitches have been performed
      int num = abs(int(coeffmodulusmap[binExprIns[i]->getLeft().getUniqueNodeId()].size() - coeffmodulusmap[binExprIns[i]->getRight().getUniqueNodeId()].size())) + 1;
      // update coeff_modulus_map
      updateCoeffModulusMap( binExprIns[i], num);
    }
  }

}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialInsertModSwitchVisitor::getCoeffModulusMap() {
  return coeffmodulusmap;
}

std::unordered_map<std::string, std::vector<seal::Modulus>> SpecialInsertModSwitchVisitor::getCoeffModulusMapVars() {
  return coeffmodulusmap_vars;
}

std::unordered_map<std::string, int> SpecialInsertModSwitchVisitor::getNoiseMap() {
  return noise_map;
};


