#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_NODEUTILS_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_NODEUTILS_H_

#include "AbstractNode.h"
#include "OpSymbEnum.h"

/// Transforms a multi-input gate taking N inputs into a sequence of binary gates.
///
/// For example, consider a N input logical-AND (&) with inputs y_1 to y_m:
///   <pre> &_{i=1}^{n} y_1, y_2, y_3, ..., y_m. </pre>
/// It is transformed by this method into the expression:
///   <pre> ((((y_1 & y_2) & y_3) ...) & y_m), </pre>
/// wherein each AND-gate only has two inputs (binary gates).

/// \param inputNodes The inputs y_1, ..., y_m that are connected to the multi-input gate. It is required that m>=2.
/// \param gateType The gate that all inputs are connected to.
/// \return A vector of AbstractNode objects of type LogicalExpr that represent the chain of LogicalExpr required to represent
/// the intended multi-input gate. The last node in inputNodes (i.e., inputNodes.back()) is always the output of this
/// chain.
std::vector<AbstractNode *> rewriteMultiInputGateToBinaryGatesChain(
    std::vector<AbstractNode *> inputNodes, LogCompOp gateType);

AbstractNode *createMultDepthBalancedTreeFromInputs(std::vector<AbstractExpr *> inputs,
                                                    OpSymbolVariant operatorType,
                                                    std::unordered_map<std::string, int> multiplicativeDepths);

AbstractNode *createMultDepthBalancedTreeFromInputs(std::vector<AbstractExpr *> inputs,
                                                    std::variant<ArithmeticOp,
                                                                 LogCompOp,
                                                                 UnaryOp> operatorType);

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_NODEUTILS_H_
