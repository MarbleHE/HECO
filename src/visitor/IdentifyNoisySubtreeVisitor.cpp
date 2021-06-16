#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"

#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialIdentifyNoisySubtreeVisitor::SpecialIdentifyNoisySubtreeVisitor(std::ostream &os, std::unordered_map<std::string, int> noise_map,
                                                   std::unordered_map<std::string, double> rel_noise_map)
    : os(os), noise_map(std::move(noise_map)), rel_noise_map(std::move(rel_noise_map)) {}

void SpecialIdentifyNoisySubtreeVisitor::visit(AbstractNode &elem) {
  // This is more of a demonstration of the Visitor template
  // since AbstractNode::toString can already output its children just fine
  // However, this PrintVisitor does technically offer the choice to indent things differently ;)

  // Get the current node's toString (without children)
  // This should hopefully be a single line, including end-of-line at the end
  std::string curNodeString = elem.toString(false);

  int leftParentNodeString = elem.countChildren();

  // Output current node at required indentation
  os << "--------------" << std::endl;
  auto result = noise_map.find(elem.getUniqueNodeId());
  auto result1 = rel_noise_map.find(elem.getUniqueNodeId());


  for (AbstractNode &c: elem) {
    c.accept(*this);
  }
}
