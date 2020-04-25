#include "MultiplicativeDepthCalculator.h"
#include "NodeUtils.h"
#include "LiteralBool.h"
#include "LogicalExpr.h"
#include "ArithmeticExpr.h"

std::vector<AbstractNode *> rewriteMultiInputGateToBinaryGatesChain(std::vector<AbstractNode *> inputNodes,
                                                                    LogCompOp gateType) {
  if (inputNodes.empty()) {
    throw std::invalid_argument("Cannot construct a 0-input logical gate!");
  }

  // if there is only one input, we need to add the "neutral element" (i.e., the element that does not change the
  // semantics of the logical expression) depending on the given LogCompOp to inputNodes
  if (inputNodes.size()==1) {
    inputNodes.push_back(OpSymb::getNeutralElement(
        OpSymbolVariant(gateType)));
  }

  // vector of resulting binary gates
  std::vector<AbstractNode *> outputNodes;

  // handle first "special" gate -> takes two inputs as specified in inputNodes
  auto it = std::begin(inputNodes);
  auto recentLexp = new LogicalExpr((*it++)->castTo<AbstractExpr>(), gateType, (*it++)->castTo<AbstractExpr>());
  outputNodes.push_back(recentLexp);

  // handle all other gates -> are connected with each other
  for (auto end = std::end(inputNodes); it!=end; ++it) {
    auto newLexp = new LogicalExpr(recentLexp, gateType, (*it)->castTo<AbstractExpr>());
    outputNodes.push_back(newLexp);
    recentLexp = newLexp;
  }
  return outputNodes;
}

AbstractNode *createMultDepthBalancedTreeFromInputs(std::vector<AbstractExpr *> inputs,
                                                    OpSymbolVariant operatorType,
                                                    std::unordered_map<std::string,
                                                                       int> multiplicativeDepths) {
  // TODO(anyone): This tree is balanced w.r.t. the multiplicative depth but does not take any FHE-specific properties
  //  e.g. scale of a ciphertext into account yet.

  if (inputs.empty()) {
    throw std::invalid_argument("Cannot rewrite a 0-input binary expression!");
  }

  // sort the inputs based on the calculated depths in increasing order (we want to connect inputs that already have a
  // large multiplicative depth to be placed as high as possible in the tree to not increase depth much more)
  auto getMultiplicativeDepth = [&multiplicativeDepths](AbstractExpr *aexp) -> int {
    auto nodeId = aexp->getUniqueNodeId();
    return (multiplicativeDepths.count(nodeId) > 0) ? multiplicativeDepths.at(nodeId) : 0;
  };
  std::sort(inputs.begin(), inputs.end(), [&](AbstractExpr *exprOne, AbstractExpr *exprTwo) {
    return getMultiplicativeDepth(exprOne) > getMultiplicativeDepth(exprTwo);
  });

  // a helper utility to create a new expression
  auto createNewExpr = [&operatorType](AbstractExpr *lhsOperand, AbstractExpr *rhsOperand) -> AbstractExpr * {
    if (std::holds_alternative<ArithmeticOp>(operatorType)) {
      return new ArithmeticExpr(lhsOperand, std::get<ArithmeticOp>(operatorType), rhsOperand);
    } else if (std::holds_alternative<LogCompOp>(operatorType)) {
      return new LogicalExpr(lhsOperand, std::get<LogCompOp>(operatorType), rhsOperand);
    } else {
      throw std::logic_error("Unsupported operator encountered in NodeUtils::createMultDepthBalancedTreeFromInputs.");
    }
  };

  do {
    std::vector<AbstractExpr *> createdExpressions;
    // connect every two inputs using a new binary expression and enqueue this new binary expression
    while (!inputs.empty() && inputs.size() >= 2) {
      auto lhsOperand = inputs.back();
      inputs.pop_back();
      auto rhsOperand = inputs.back();
      inputs.pop_back();
      createdExpressions.push_back(createNewExpr(lhsOperand, rhsOperand));
    }
    // add any remaining, non connected inputs to createdExpressions (they need to be connected later)
    createdExpressions.insert(createdExpressions.end(), inputs.begin(), inputs.end());
    // reverse the createdExpressions vector because createdExpressions.back() now contains the var with highest mdepth
    // but we want to proceed connecting those with the lowest multiplicative depth first
    std::reverse(createdExpressions.begin(), createdExpressions.end());
    inputs = std::move(createdExpressions);
  } while (inputs.size() > 1);  // stop if there is only one binary expression left (root of created subtree)

  // inputs now contains only one AbstractExpr: this is the root of the newly created tree
  return inputs.front();
}

AbstractNode *createMultDepthBalancedTreeFromInputs(std::vector<AbstractExpr *> inputs,
                                                    std::variant<ArithmeticOp,
                                                                 LogCompOp,
                                                                 UnaryOp> operatorType) {
  return createMultDepthBalancedTreeFromInputs(inputs, operatorType,
                                               std::unordered_map<std::string, int>());
}
