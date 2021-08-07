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
      for (int i = leftIndex; i >= (leftIndex - (abs(diff) + 1)); i--) {
        sum += coeffmodulusmap[elem.getLeft().getUniqueNodeId()][i].bit_count();
      }
      // if the spent noise budgets for the left and right operators are sufficiently large, we suggest a modswitch insertion procedure
      if ((sum < spentNoiseBudgetLeft) && spentNoiseBudgetRight > coeffmodulusmap[elem.getRight().getUniqueNodeId()][rightIndex].bit_count()) {
        std::cout << "Suggesting insertion of modswitch(es) applied to operands of node " << elem.getUniqueNodeId() << std::endl;
      }
    }
    // if diff < 0 then the left operand has fewer primes remaining in the coeffmodulus chain
    else if (diff <0) {
      int sum = 0;
      for (int i = rightIndex; i >= (rightIndex - (abs(diff) + 1)); i--) {
        sum += coeffmodulusmap[elem.getRight().getUniqueNodeId()][i].bit_count();
      }
      // if the spent noise budgets for the left and right operators are sufficiently large, we suggest a modswitch insertion procedure
      if ((sum < spentNoiseBudgetRight) && spentNoiseBudgetLeft > coeffmodulusmap[elem.getLeft().getUniqueNodeId()][leftIndex].bit_count()) {
        std::cout << "Suggesting insertion of modswitch(es) applied to operands of node " << elem.getUniqueNodeId() << std::endl;
      }
    }
    // if diff = 0, check if a single modswitch after both operands can be inserted
    else {
      if ((spentNoiseBudgetLeft > coeffmodulusmap[elem.getLeft().getUniqueNodeId()][leftIndex].bit_count()) &&
            (spentNoiseBudgetRight > coeffmodulusmap[elem.getRight().getUniqueNodeId()][rightIndex].bit_count())) {
        std::cout << "Suggesting insertion of modswitch(es) applied to operands of node " << elem.getUniqueNodeId() << std::endl;
      }
    }
  }
}

std::unique_ptr<AbstractNode> SpecialInsertModSwitchVisitor::rewriteAst(std::unique_ptr<AbstractNode> &&ast_in) {

}

std::unique_ptr<BinaryExpression> SpecialInsertModSwitchVisitor::getModSwitchNode() const{
  return nullptr;
};


