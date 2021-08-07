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



}

std::unique_ptr<AbstractNode> SpecialInsertModSwitchVisitor::rewriteAst(std::unique_ptr<AbstractNode> &&ast_in) {

}

std::unique_ptr<BinaryExpression> SpecialInsertModSwitchVisitor::getModSwitchNode() const{
  return nullptr;
};


