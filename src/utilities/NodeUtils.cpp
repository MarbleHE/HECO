#include "NodeUtils.h"
#include "LiteralBool.h"
#include "LogicalExpr.h"
#include "ArithmeticExpr.h"

std::vector<AbstractNode *> rewriteMultiInputGateToBinaryGatesChain(std::vector<AbstractNode *> inputNodes,
                                                                    OpSymb::LogCompOp gateType) {
  if (inputNodes.empty()) {
    throw std::invalid_argument("Cannot construct a 0-input logical gate!");
  }

  // if there is only one input, we need to add the "neutral element" (i.e., the element that does not change the
  // semantics of the logical expression) depending on the given LogCompOp to inputNodes
  if (inputNodes.size()==1) {
    // TODO(pjattke): Replace this by call to OpSymb::getIdentityElement(...)
    if (gateType==OpSymb::LogCompOp::logicalXor) {
      // inputNodes[0] XOR false
      inputNodes.push_back(new LiteralBool(false));
    } else if (gateType==OpSymb::LogCompOp::logicalAnd) {
      // inputNodes[0] AND true
      inputNodes.push_back(new LiteralBool(true));
    } else {
      throw std::runtime_error(
          "Method rewriteMultiInputGateToBinaryGatesChain currently supports 1-input gates of type logical-XOR "
          "or logical-AND only.");
    }
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

std::vector<AbstractNode *> rewriteMultiInputBinaryExpressionToBinaryExpressionGatesChain(std::vector<AbstractExpr *> inputs,
                                                                                          std::variant<OpSymb::ArithmeticOp,
                                                                                                       OpSymb::LogCompOp,
                                                                                                       OpSymb::UnaryOp> operatorType) {
  if (inputs.empty()) {
    throw std::invalid_argument("Cannot rewrite a 0-input binary expression!");
  }

  // if there is only one input, we need to add the "neutral element" (i.e., the element that does not change the
  // semantics of the expression) depending on the given operator
  if (inputs.size()==1) {
    inputs.push_back(OpSymb::getIdentityElement(operatorType));
  }

  auto createNewExpr = [&operatorType](AbstractExpr *lhsOperand, AbstractExpr *rhsOperand) -> AbstractExpr * {
    if (std::holds_alternative<OpSymb::ArithmeticOp>(operatorType)) {
      return new ArithmeticExpr(lhsOperand, std::get<OpSymb::ArithmeticOp>(operatorType), rhsOperand);
    } else if (std::holds_alternative<OpSymb::LogCompOp>(operatorType)) {
      return new LogicalExpr(lhsOperand, std::get<OpSymb::LogCompOp>(operatorType), rhsOperand);
    } else {
      throw std::logic_error("");
    }
  };

  // vector of resulting binary gates
  std::vector<AbstractNode *> outputNodes;

  // handle first "special" gate -> takes two inputs as specified in inputNodes
  auto it = std::begin(inputs);
  auto recentBexp = createNewExpr((*it++)->castTo<AbstractExpr>(), (*it++)->castTo<AbstractExpr>());
  outputNodes.push_back(recentBexp);

  // handle all other gates -> are connected with each other
  for (auto end = std::end(inputs); it!=end; ++it) {
    auto newLexp = createNewExpr(recentBexp, (*it)->castTo<AbstractExpr>());
    outputNodes.push_back(newLexp);
    recentBexp = newLexp;
  }
  return outputNodes;
}
