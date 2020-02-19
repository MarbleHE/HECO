#include "Ast.h"

#include <variant>
#include <iostream>
#include <set>
#include <queue>
#include <EvaluationVisitor.h>
#include "AbstractLiteral.h"
#include "Variable.h"
#include "Function.h"
#include "Return.h"
#include "LogicalExpr.h"

Ast::Ast(AbstractNode *rootNode) : rootNode(rootNode) {}

Ast::Ast() {
  rootNode = nullptr;
}

AbstractNode *Ast::setRootNode(AbstractNode *node) {
  this->rootNode = node;
  return this->rootNode;
}

AbstractNode *Ast::getRootNode() const {
  return rootNode;
}

void Ast::accept(Visitor &v) {
  v.visit(*this);
}

Ast::~Ast() {
  delete rootNode;
}

std::vector<AbstractLiteral *>
Ast::evaluateCircuit(const std::unordered_map<std::string, AbstractLiteral *> &paramValues, bool printResult) {
  // ensure that circuit is not reversed
  if (isReversed()) {
    throw std::invalid_argument(
        "Cannot evaluate reversed circuit. Ensure that all edges of the circuit are reversed, then try again.");
  }

  // go through all nodes and collect all identifiers of Variable nodes
  std::set<std::string> varIdentifiersReqValue;
  auto isVariableNode = [](AbstractNode *node) { return dynamic_cast<Variable *>(node)!=nullptr; };
  for (auto &node : getAllNodes(isVariableNode)) {
    varIdentifiersReqValue.insert(node->castTo<Variable>()->getIdentifier());
  }

  // Remove those variable identifiers from variableIdentifiersRequiringValue that are defined using a VarDecl in
  // the circuit -> those do not need a value.
  auto isVarDeclNode = [](AbstractNode *node) { return dynamic_cast<VarDecl *>(node)!=nullptr; };
  for (auto &node : getAllNodes(isVarDeclNode)) {
    varIdentifiersReqValue.erase(
        std::find(varIdentifiersReqValue.begin(),
                  varIdentifiersReqValue.end(),
                  node->castTo<VarDecl>()->getVarTargetIdentifier()));
  }

  // ensure that the provided number of parameters equals the number of required ones
  if (paramValues.size()!=varIdentifiersReqValue.size())
    throw std::invalid_argument("AST evaluation requires parameter value for all variables!");

  // Make sure that all variables collected previously have a defined value.
  // Note: On contrary to the sanity checks in evaluateAst(...), we cannot check the datatypes as we do not have
  // a FunctionParameter with datatype information as used by Function objects. Therefore, we limit the check to the
  // presence of any value.
  for (auto &var : varIdentifiersReqValue) {
    if (paramValues.find(var)==paramValues.end()) {
      std::stringstream ss;
      ss << "No parameter value was given for Variable ";
      ss << var << ".";
      throw std::logic_error(ss.str());
    }
  }

  // create a new EvaluationVisitor instance, perform evaluation, return evaluation results
  EvaluationVisitor ev(paramValues);
  ev.setFlagPrintResult(printResult);
  ev.visit(*this);
  return ev.getResults();
}

std::vector<AbstractLiteral *>
Ast::evaluateAst(const std::unordered_map<std::string, AbstractLiteral *> &paramValues, bool printResult) {
  // ensure that the root node is a Function
  auto func = dynamic_cast<Function *>(this->getRootNode());
  if (!func) {
    throw std::logic_error(
        "AST evaluation only supported for 'Function' root node! Consider using evaluateCircuit(...) instead.");
  }

  // A helper function that returns the variable's value of paramValues, if the variable exists in the map, otherwise
  // returns a nullptr.
  auto getVarValue = [&paramValues](const std::string &variableIdentifier) -> AbstractLiteral * {
    auto it = paramValues.find(variableIdentifier);
    return (it==paramValues.end()) ? nullptr : it->second;
  };
  // A helper function that returns True if a variable identifier (std::string) exists in the paramValues map, i.e, a
  // there is a value for the given variable. Otherwise returns False.
  auto hasVarValue = [&paramValues](Variable *var) -> bool { return paramValues.count(var->getIdentifier()) > 0; };

  // ensure that the provided number of parameters equals the number of required ones
  if (paramValues.size()!=func->getParameters().size())
    throw std::invalid_argument("AST evaluation requires parameter value for all parameters!");

  // make sure that all parameters specified by the function have a defined value
  for (const auto &fp : func->getParameters()) {
    // check if FunctionParameter is a variable (can also be a hard-coded value, e.g., a LiteralInt)
    if (auto var = dynamic_cast<Variable *>(fp->getValue())) {
      // throw an error if variable value for var is not defined
      if (!hasVarValue(var)) {
        throw std::invalid_argument("AST evaluation requires parameter value for parameter " + var->getIdentifier());
      }
      // throw an error if type of given parameter value and type of expected value do not match
      if (!getVarValue(var->getIdentifier())->supportsDatatype(*fp->getDatatype())) {
        std::stringstream ss;
        ss << "AST evaluation cannot proceed because datatype of given parameter and expected datatype differs:\n";
        ss << "Variable " << var->getIdentifier() << " ";
        ss << "(value: " << *getVarValue(var->getIdentifier()) << ")";
        ss << " is not of type ";
        ss << fp->getDatatype()->toString() << ".";
        throw std::invalid_argument(ss.str());
      }
    }
  }

  // create a new EvaluationVisitor instance, perform evaluation, return evaluation results
  EvaluationVisitor ev(paramValues);
  ev.setFlagPrintResult(printResult);
  ev.visit(*this);
  return ev.getResults();
}

bool Ast::isReversed() const {
  int sum = 0;
  auto allNodes = getAllNodes();
  for (auto &node : allNodes) { if (node->isReversed) sum++; }

  // check that we are in a consistent state, i.e., either all of the nodes have reversed edges or none of them
  bool allReversed = (sum==allNodes.size());
  bool noneReversed = (sum==0);
  if (!allReversed && !noneReversed) {
    throw std::runtime_error("Inconsistent state! AST consists of some nodes that have reversed edges.");
  } else {
    return allReversed;
  }
}

Ast::Ast(const Ast &otherAst, bool keepOriginalUniqueNodeId) : rootNode(nullptr) {
  this->setRootNode(otherAst.getRootNode()->clone(keepOriginalUniqueNodeId));
}

Ast::Ast(const Ast &otherAst) : Ast(otherAst, false) {}

bool Ast::isValidCircuit() {
  std::set<AbstractNode *> allAstNodes = getAllNodes();
  auto supportCircuitMode = [](AbstractNode *n) { return n->supportsCircuitMode(); };
  return std::all_of(allAstNodes.begin(), allAstNodes.end(), supportCircuitMode);
}

void Ast::reverseEdges() {
  for (auto &node : getAllNodes()) node->swapChildrenParents();
}

std::set<AbstractNode *> Ast::getAllNodes() const {
  return getAllNodes(nullptr);
}

std::set<AbstractNode *> Ast::getAllNodes(const std::function<bool(AbstractNode *)> &predicate) const {
  // the result set of all nodes in the AST
  std::set<AbstractNode *> allNodes{};
  // the nodes still required to be processed
  std::queue<AbstractNode *> nodesToCheck{{getRootNode()}};
  // continue while there are still unprocessed nodes
  while (!nodesToCheck.empty()) {
    // deque next node to process
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    // skip curNode if its a nullptr
    if (curNode==nullptr) continue;
    // if (no predicate is set) OR (predicate is set AND node fulfills predicate) -> add node to result set
    if (predicate==nullptr || predicate(curNode)) allNodes.insert(curNode);
    // depending on the status of the node, enqueue the next nodes
    auto nextNodesToConsider = curNode->isReversed ? curNode->getParentsNonNull() : curNode->getChildrenNonNull();
    for (auto &c : nextNodesToConsider) nodesToCheck.push(c);
  }
  return allNodes;
}

void Ast::deleteNode(AbstractNode **node, bool deleteSubtreeRecursively) {
  AbstractNode *nodePtr = *node;
  nodePtr->getUniqueNodeId();

  // handle the node's children
  if (deleteSubtreeRecursively) {
    // if deleteSubtreeRecursively is set, we need to delete all children first
    for (auto &c : nodePtr->getChildrenNonNull()) {
      c->getUniqueNodeId();
      deleteNode(&c, true);
    }
  } else if (!nodePtr->getChildrenNonNull().empty()) {
    // if deleteSubtreesRecursively is not set but there are children, we cannot continue.
    // probably the user's intention was not to delete the whole subtree.
    throw std::logic_error("Cannot remove node (" + nodePtr->getUniqueNodeId()
                               + ") because node has children but deleteSubtreeRecursively is not set (false).");
  }

  // remove AST's root node if this node is the AST's root node
  if (nodePtr==this->getRootNode()) this->setRootNode(nullptr);

  // first isolate the node from its parents, then deallocate the heap memory
  nodePtr->isolateNode();
  delete nodePtr;

  // "clear" the passed pointer to avoid the further use of this deleted node
  *node = nullptr;
}

