#include "ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/VariableDeclaration.h"

#include <deque>

void SpecialControlFlowGraphVisitor::checkEntrypoint(AbstractNode &node) {
  // if this is not the first visited node, then there's no need for this check
  if (!nodes.empty()) return;

  // if this is the first visited node ---
  // make sure that CFGV was called on a Function, Block, If, or For node
  auto nodeAsFunction = dynamic_cast<Function *>(&node);
  auto nodeAsBlock = dynamic_cast<Block *>(&node);
  auto nodeAsIf = dynamic_cast<If *>(&node);
  auto nodeAsFor = dynamic_cast<For *>(&node);

  if (nodeAsFunction==nullptr && nodeAsBlock==nullptr && nodeAsIf==nullptr && nodeAsFor==nullptr) {
    throw std::runtime_error(
        "Cannot run ControlFlowGraphVisitor on given node (" + node.getUniqueNodeId() + "). " +
            "Visitor requires a Function, Block, For, or If AST node as (root) input.");
  }
}

void SpecialControlFlowGraphVisitor::visit(Assignment &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Assignment (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);

  std::string identifier;
  if (auto variable = dynamic_cast<Variable *>(&node.getTarget())) {
    identifier = variable->getIdentifier();
  } else if (auto indexAccess = dynamic_cast<IndexAccess *>(&node.getTarget())) {
    // TODO implement me: must recursively go tree down and retrieve all variable identifiers
  }

  // visit the right-hand side of the assignment
  node.getValue().accept(*this);

  if (getCurrentScope().identifierExists(identifier) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(identifier), VariableAccessType::WRITE);
  }

  storeAccessedVariables(graphNode);
}

void SpecialControlFlowGraphVisitor::visit(Block &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Block (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(graphNode);
}

////////////////////////////////////
///////////// For Statement ////////
////////////////////////////////////
// CFG Graph
//
//        Initializer Stmt. 1
//                 │
//                 ▼
//    ┌── Initializer Stmt. N
//    │
//    │
//    │      Body Stmt. 1    ◀─┐
//    │            │           │
//    │            ▼           │
//    │      Body Stmt. N      │
//    │            │           │
//    │            │           │
//    │            │           │
//    │            ▼           │
//    │     Update Stmt. 1     │
//    │            │           │
//    │            ▼           │
//    │     Update Stmt. N     │
//    │            │           │
//    │            │           │ condition
//    │            ▼           │  == true
//    └─────▶ Condition*  ─────┘
//                 │
//                 │   condition
//                 │   == false
//                 ▼
//            Next Stmt.
//
// (*)Note: Although the Condition is not a AbstractStatement in our AST model, we include it here as we need the
// information which variables are accessed.
void SpecialControlFlowGraphVisitor::visit(For &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting For (" << node.getUniqueNodeId() << ")" << std::endl;
  [[maybe_unused]] auto &graphNode = createGraphNodeAndAppendToCfg(node);

  ScopedVisitor::enterScope(node);

  // initializer (e.g., int i = 0;)
  // we need to use the visitChildren method here as we do not want to have the initializer's Block to be included in
  // the CFG as separate GraphNode as this would cause opening a new scope
  ScopedVisitor::visitChildren(node.getInitializer());
  auto lastStatementInInitializer = lastCreatedNodes;

  // condition expression (e.g., i <= N)
  GraphNode &gNodeCondition = createGraphNodeAndAppendToCfg(node.getCondition());
  node.getCondition().accept(*this);
  storeAccessedVariables(gNodeCondition);
  auto lastStatementCondition = lastCreatedNodes;

  // body (e.g., For (-; -; -) { body statements ... })
  node.getBody().accept(*this);
  auto lastStatementInBody = lastCreatedNodes;

  // update statement (e.g., i=i+1;)
  // we need to use the visitChildren method here as we do not want to have the update's Block to be included in
  // the CFG as separate GraphNode as this would cause opening a new scope
  ScopedVisitor::visitChildren(node.getUpdate());
  auto lastStatementInUpdate = lastCreatedNodes;

  ScopedVisitor::exitScope(node);

  // edge: update statement -> condition
  auto firstConditionStatement = lastStatementInInitializer.front().get().getControlFlowGraph().getChildren().front();
  if (!lastStatementInUpdate.empty()) {
    for (auto &updateStatement : lastStatementInUpdate) {
      // create an edge in CFG from last update statement to first statement in condition as the condition is checked
      // after executing the update statement of a loop iteration
      updateStatement.get().getControlFlowGraph().addChild(firstConditionStatement);
    }
  } else {
    // if there is no update statement, create an edge from last body statement to condition
    lastStatementInBody.front().get().getControlFlowGraph().addChild(firstConditionStatement);
  }

  // edge: condition -> next statement after If statement
  lastCreatedNodes = lastStatementCondition;
}

void SpecialControlFlowGraphVisitor::visit(Function &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Function (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);

  ScopedVisitor::enterScope(node);

  // visit the parameters and remember that they have been declared (is equal to writing to a variable)
  for (auto &param : node.getParameters()) {
    param.get().accept(*this);
  }
  storeAccessedVariables(graphNode);

  // access the function's body
  node.getBody().accept(*this);

  ScopedVisitor::exitScope(node);
}

void SpecialControlFlowGraphVisitor::visit(FunctionParameter &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  ScopedVisitor::visit(node);
  std::cout << "Visiting FunctionParameter (" << node.getUniqueNodeId() << ")" << std::endl;
  if (getCurrentScope().identifierExists(node.getIdentifier()) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(node.getIdentifier()), VariableAccessType::WRITE);
  }
}

void SpecialControlFlowGraphVisitor::visit(If &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting If (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  auto lastStatementIf = lastCreatedNodes;

  ScopedVisitor::enterScope(node);

  // condition
  node.getCondition().accept(*this);
  storeAccessedVariables(graphNode);

  // then branch: connect this statement with then branch
  node.getThenBranch().accept(*this);
  auto lastStatementThenBranch = lastCreatedNodes;

  // if existing, connect If statement with Else block
  if (node.hasElseBranch()) {
    lastCreatedNodes = lastStatementIf;
    // else branch
    node.getElseBranch().accept(*this);
  }

  ScopedVisitor::exitScope(node);
}

void SpecialControlFlowGraphVisitor::visit(Return &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Return (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(graphNode);
}

void SpecialControlFlowGraphVisitor::visit(VariableDeclaration &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting VariableDeclaration (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);

  // We do not use ScopedVisitor::visit here as this would visit all children, including the Variable on the left-hand
  // side of the VariableDeclaration. This would register this variable as having been read, although it is written to
  // this variable.

  if (node.hasValue()) {
    node.getValue().accept(*this);
  }

  getCurrentScope().addIdentifier(node.getTarget().getIdentifier());

  if (getCurrentScope().identifierExists(node.getTarget().getIdentifier()) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(node.getTarget().getIdentifier()),
                       VariableAccessType::WRITE);
  }

  storeAccessedVariables(graphNode);
}

GraphNode &SpecialControlFlowGraphVisitor::createGraphNodeAndAppendToCfg(AbstractNode &astNode) {
  return createGraphNodeAndAppendToCfg(astNode, lastCreatedNodes);
}

GraphNode &SpecialControlFlowGraphVisitor::createGraphNodeAndAppendToCfg(
    AbstractNode &statement,
    const std::vector<std::reference_wrapper<GraphNode>> &parentNodes) {

  // create a new GraphNode for the given statement
  auto graphNodeUniquePtr = std::make_unique<GraphNode>(statement);
  GraphNode &graphNodeRef = *graphNodeUniquePtr;

  // add this node as child to all parents passed in parentNodes vector
  for (auto &p : parentNodes) {
    p.get().getControlFlowGraph().addChild(graphNodeRef);
  }

  // update the lastCreatedNodes vector
  lastCreatedNodes.clear();
  lastCreatedNodes.emplace_back(graphNodeRef);

  // store this node so that we can delete it during the visitor's destruction
  // NOTE: After this statement graphNodeUniquePtr is not accessible anymore!
  nodes.emplace_back(std::move(graphNodeUniquePtr));

  return graphNodeRef;
}

void SpecialControlFlowGraphVisitor::storeAccessedVariables(GraphNode &graphNode) {
  // expects that exactly one GraphNode was created before calling this method
  if (!nodes.empty()) {
    graphNode.setAccessedVariables(std::move(variableAccesses));
  } else {
    throw std::runtime_error("Cannot assign accessed variables to GraphNode because no GraphNode exists.");
  }
}

void SpecialControlFlowGraphVisitor::visit(Variable &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  ScopedVisitor::visit(node);
  if (getCurrentScope().identifierExists(node.getIdentifier()) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(node.getIdentifier()), VariableAccessType::READ);
  }
}

void SpecialControlFlowGraphVisitor::markVariableAccess(const ScopedIdentifier &scopedIdentifier,
                                                        VariableAccessType accessType) {
  if (variableAccesses.count(scopedIdentifier) > 0) {
    // we potentially need to merge the existing access type and the new one
    if (variableAccesses.at(scopedIdentifier)==accessType) {
      // no need to merge types as both types are the same
      return;
    } else {
      // as they are different and there are only two types (read, write), both (read+write) must be set
      variableAccesses[scopedIdentifier] = VariableAccessType::READ_AND_WRITE;
    }
  } else {
    // we can just add the new variable access
    variableAccesses[scopedIdentifier] = accessType;
  }
}

GraphNode &SpecialControlFlowGraphVisitor::getRootNode() {
  return *nodes.front().get();
}

const GraphNode &SpecialControlFlowGraphVisitor::getRootNode() const {
  return *nodes.front().get();
}

SpecialControlFlowGraphVisitor::SpecialControlFlowGraphVisitor(bool ignoreNonDeclaredVariables)
    : ignoreNonDeclaredVariables(ignoreNonDeclaredVariables) {
}

SpecialControlFlowGraphVisitor::SpecialControlFlowGraphVisitor(
    std::vector<std::string> &alreadyDeclaredVariables) {
  ScopedVisitor::setPredeclaredVariables(alreadyDeclaredVariables);
}

void SpecialControlFlowGraphVisitor::buildDataflowGraph() {
  // TODO: Improve naming of variables

  // =================
  // STEP 1: Distribute knowledge about variable writes
  // Traverse the graph and store for each graph node where (i.e., at which node) all of the variables seen so far were
  // written the last time with respect to the program's execution flow (control flow graph). This works by propagating
  // the knowledge about variable writes through the CFG and updating it with the local information (from the current 
  // graph node).
  // =================

  typedef std::unordered_map<ScopedIdentifier,
                             std::vector<std::reference_wrapper<GraphNode>>,
                             ScopedIdentifierHashFunction,
                             std::equal_to<>> IdentifierGraphNodeMap;

  // temporary map (to be used by currently processed node) to store where a variable was last written
  IdentifierGraphNodeMap varsLastWritten;

  // a map that contains for each AST node (key: unique ID) associated with a GraphNode a map describing at which node
  // all variables (value->key) seen so far were written at the last time (value->value); as the latter can be at
  // multiple nodes this is modeled as vector
  std::unordered_map<std::string, IdentifierGraphNodeMap> nodeToVarLastWrittenMapping;

  // a map of all already processed nodes used to detect cycles
  std::unordered_map<std::string, std::reference_wrapper<GraphNode>> processedNodes;

  // a queue of nodes to be processed next
  std::deque<std::reference_wrapper<GraphNode>> nextNodesToVisit({std::ref(getRootNode())});

  // iterate over all nodes that are reachable from the CFG's root node
  while (!nextNodesToVisit.empty()) {
    // reset temporary structure before next iteration
    varsLastWritten.clear();

    // get node to be processed in this iteration
    auto currentNode = nextNodesToVisit.front();
    nextNodesToVisit.pop_front();

    // extract required information from node
    auto currentNode_id = currentNode.get().getAstNode().getUniqueNodeId();
    auto currentNode_parentNodes = currentNode.get().getControlFlowGraph().getParents();

    // joint points a CFG nodes with multiple incoming edges (e.g., statement after a [return-free] If-Else statement)
    bool currentNode_isJointPoint = currentNode_parentNodes.size() > 1;

    // iterate over currentNode's parents and collect their knowledge about variable writes
    for (auto &pNode : currentNode_parentNodes) {
      auto pNode_id = pNode.get().getAstNode().getUniqueNodeId();
      for (auto &[scopedIdentifier, nodesWritingToVariable] : nodeToVarLastWrittenMapping.at(pNode_id)) {
        // either add the nodes that refer to the variable (then) in case that this variable is already known [merging],
        // or (else) create a new vector by assigning/copying the vector of referenced nodes [replacing]
        if (varsLastWritten.count(scopedIdentifier) > 0) {
          auto &vec = varsLastWritten.at(scopedIdentifier);
          vec.insert(vec.end(), nodesWritingToVariable.begin(), nodesWritingToVariable.end());
        } else {
          varsLastWritten[scopedIdentifier] = nodesWritingToVariable;
        }
      }
    }

    for (auto &scopedId : currentNode.get()
        .getVariableAccessesByType({VariableAccessType::WRITE, VariableAccessType::READ_AND_WRITE})) {
      if (!currentNode_isJointPoint && varsLastWritten.count(scopedId) > 0) varsLastWritten.at(scopedId).clear();
      varsLastWritten[scopedId].push_back(currentNode);
    }

    // compare varLastWritten with information in nodeToVarLastWrittenMapping to see whether there were any changes such
    // that the newly collected knowledge needs to be distributed to the children nodes again -> revisit required
    // (check will be evaluated only if the node was visited once before, see condition in visitingChildrenRequired)
    auto collectedWrittenVarsChanged = [&]() -> bool {
      auto mp = nodeToVarLastWrittenMapping.at(currentNode_id);
      return std::any_of(varsLastWritten.begin(), varsLastWritten.end(), [&mp](const auto &mapEntry) {
        return mp.count(mapEntry.first)==0 ||  // a variable that was not tracked before
            (mp.at(mapEntry.first).size()
                !=mapEntry.second.size());  // a variable for which new write nodes were determined
      });
    };

    // condition that decides whether children must be enqueued / visited next
    bool visitingChildrenRequired = (
        processedNodes.count(currentNode_id)==0)  // node was not visited yet
        || (nodeToVarLastWrittenMapping.count(currentNode_id) > 0
            && collectedWrittenVarsChanged()); // information changed


    // attach the collected write information to this node, for that it is required to check if there is already
    // existing information in nodeToVarLastWrittenMapping about this node (then append) or not (then move/overwrite)
    if (nodeToVarLastWrittenMapping.count(currentNode_id)==0) {
      // simply move the collected nodes in varLastWritten to nodeToVarLastWrittenMapping
      nodeToVarLastWrittenMapping[currentNode_id] = std::move(varsLastWritten);
    } else {
      // merge the nodes already existing in nodeToVarLastWrittenMapping with those newly collected
      for (auto &[varIdentifier, gNodeSet] : varsLastWritten) {
        auto set = varsLastWritten.at(varIdentifier);
        auto vec = nodeToVarLastWrittenMapping.at(currentNode_id).at(varIdentifier);
        vec.insert(vec.end(), set.begin(), set.end());
      }
    }

    // check if we need to visit the current node's children (else) or we can continue with the next node (then)
    if (!visitingChildrenRequired) {
      continue;
    } else {
      // enqueue children of current node: those are the ones to visit next
      auto nextNodes = currentNode.get().getControlFlowGraph().getChildren();
      nextNodesToVisit.insert(nextNodesToVisit.end(), nextNodes.begin(), nextNodes.end());
      // mark this node as visited to not visit it again
      processedNodes.emplace(currentNode_id, currentNode);
    }

  } // end:  while (!nextNodesToVisit.empty())

  // =================
  // STEP 2:
  // Traverse all graph nodes that have variable reads and add an edge to the last location where the respective
  // variable was written last.
  // =================

  // for each node that was visited in the CFG
  for (auto &[k, v] : processedNodes) {

    // retrieve all variables that were read
    for (auto &scopedIdentifier : v.get()
        .getVariableAccessesByType({VariableAccessType::READ, VariableAccessType::READ_AND_WRITE})) {

      if (v.get().getAccessedVariables().at(scopedIdentifier)==VariableAccessType::READ_AND_WRITE) {
        for (auto &parentNode : v.get().getControlFlowGraph().getParents()) {

          auto mapEntry = nodeToVarLastWrittenMapping.at(parentNode.get().getAstNode().getUniqueNodeId());

          if (mapEntry.count(scopedIdentifier)==0) continue;

          auto vectorOfWrites = mapEntry.at(scopedIdentifier);
          for (auto &edgeSrc : vectorOfWrites) edgeSrc.get().getDataFlowGraph().addChild(v);

        }
      } else {
        for (auto &writeNodes : nodeToVarLastWrittenMapping.at(v.get().getAstNode().getUniqueNodeId())
            .at(scopedIdentifier)) {
          writeNodes.get().getDataFlowGraph().addChild(v);
        }
      }

    }

  }

}
