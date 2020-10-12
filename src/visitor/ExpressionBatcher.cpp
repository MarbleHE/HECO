#include <queue>
#include <iostream>
#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/ExpressionBatcher.h"

////////////////////////////////////////////
////         ExpressionBatching         ////
////////////////////////////////////////////

class ComputationOperator {
 private:
  Operator op;
  bool batched = false;
 public:
  ComputationOperator(const Operator &op, bool batched = false) : op(op), batched(batched) {}
  bool isBatched() {return batched;}
  Operator getOp() {return op;}
};

/// Invariant (not enforced by class): if computationOperator::batched == true, then children.size() must be 2
class TreeNode {
 private:
  TreeNode *parent = nullptr;
  std::vector<std::unique_ptr<TreeNode>> children;

 public:
  virtual ~TreeNode() = default;
  TreeNode() = default;
  TreeNode(TreeNode *parent, std::vector<std::unique_ptr<TreeNode>> &&children) :
      parent(parent), children(std::move(children)) {}
  TreeNode(const TreeNode &other) = delete;
  TreeNode(TreeNode &&other) noexcept = default;
  TreeNode &operator=(const TreeNode &other) = delete;
  TreeNode &operator=(TreeNode &&other) = default;

  void addChild(std::unique_ptr<TreeNode> &&newChild) {
    newChild->parent = this;
    for (auto &p : children) {
      if (p==nullptr) {
        p = std::move(newChild);
        return;
      }
    }
    throw std::runtime_error("Did not find empty slot to insert child into.");
  }
  TreeNode *getParent() { return parent; }
  std::vector<std::unique_ptr<TreeNode>> &getChildren() { return children; }
  [[nodiscard]] size_t getExpectedNumberOfChildren() const { return children.size(); }
  friend TreeNode *insertNodeIntoTree(std::unique_ptr<TreeNode> &&newNode, TreeNode *curTreeNode);
};

class ComputationNode : public TreeNode {
 private:
  ComputationOperator
      computationOperator = ComputationOperator(Operator(ArithmeticOp::FHE_ADDITION)); //Dummy since no () ctor
 public:
  ~ComputationNode() override = default;
  ComputationNode() = default;
  ComputationNode(TreeNode *parent,
                  std::vector<std::unique_ptr<TreeNode>> &&children,
                  ComputationOperator computationOperator) : TreeNode(parent, std::move(children)),
                                                             computationOperator(computationOperator) {}
  ComputationNode(const ComputationNode &other) = delete;
  ComputationNode(ComputationNode &&other) noexcept = default;
  ComputationNode &operator=(const ComputationNode &other) = delete;
  ComputationNode &operator=(ComputationNode &&other) = default;

  ComputationOperator &getComputationOperator() { return computationOperator; }
};

class ValueNode : public TreeNode {
 private:
  AbstractNode *value;
 public:
  ~ValueNode() override = default;
  ValueNode() = default;
  ValueNode(TreeNode *parent, std::vector<std::unique_ptr<TreeNode>> &&children, AbstractNode &value) :
      TreeNode(parent, std::move(children)),
      value(&value) {}
  ValueNode(const ValueNode &other) = delete;
  ValueNode(ValueNode &&other) noexcept = default;
  ValueNode &operator=(const ValueNode &other) = delete;
  ValueNode &operator=(ValueNode &&other) = default;

};
bool operator==(const std::unique_ptr<TreeNode> &sp, const TreeNode *const p) { return sp.get()==p; }

///
/// \param newNode
/// \param curTreeNode
/// \return New curTreeNode pointer, if it needed to be changed
TreeNode *insertNodeIntoTree(std::unique_ptr<TreeNode> &&newNode, TreeNode *curTreeNode) {
  // Check if we can insert into the currentTreeNode
  for (auto &p : curTreeNode->children) {
    if (p==nullptr) {
      newNode->parent = curTreeNode;
      p = std::move(newNode);
      return curTreeNode;
    }
  }

  // If we got here, we can't add it to curTreeNode so go to next node in the current level
  if (!curTreeNode->parent) {
    // We are at the root, so no siblings => end of the current level
  } else {
    std::vector<std::unique_ptr<TreeNode>> &v = curTreeNode->getParent()->getChildren();
    auto it = std::find(v.begin(), v.end(), curTreeNode);
    if (it==v.end()) {
      throw std::runtime_error("Inconsistency in Tree: Could not find node in children list of its parent.");
    }
    // advance to next node
    ++it;
    if (it!=v.end()) {
      // recursively try to insert into the next node's children
      return insertNodeIntoTree(std::move(newNode), it->get());
    } else {
      // We are at the end of the current level
    }
  }

  // Try to go down one level
  for(auto &c: curTreeNode->children) {
    if(c) {
      return insertNodeIntoTree(std::move(newNode), c.get());
    }
  }
  // If we get here, the current node had no non-null children
  throw std::runtime_error("Cannot insert a new node since the tree ended unexpectedly.");
}

std::unique_ptr<TreeNode> SpecialExpressionBatcher::batchExpression(AbstractExpression &expr, BatchingConstraint) {
  //TODO: Introduce cut-off for batching when not enough leaf nodes!

  /// Holds nodes of the current level
  std::deque<std::reference_wrapper<AbstractNode>> currentLevelNodes({expr});

  /// Holds nodes of the next level
  std::deque<std::reference_wrapper<AbstractNode>> nextLevelNodes;

  std::unique_ptr<TreeNode> computationTree = nullptr;
  TreeNode *curTreeNode = nullptr;

  /// Each iteration of the loop corresponds to handling one level of the expression tree
  while (!currentLevelNodes.empty()) {

    // Get the next node to process
    auto &curAbstractNode = currentLevelNodes.front().get();
    currentLevelNodes.pop_front();

    // Process the AbstractExpression node, generating a TreeNode for it

    /// TreeNode for the current AST node
    std::unique_ptr<TreeNode> treeNode = nullptr;

    if (auto binaryExpression = dynamic_cast<BinaryExpression *>(&curAbstractNode)) {
      std::vector<std::unique_ptr<TreeNode>> children;
      children.emplace_back(nullptr);
      children.emplace_back(nullptr);
      treeNode = std::make_unique<ComputationNode>(
          nullptr,
          std::move(children),
          ComputationOperator(binaryExpression->getOperator(), true));
    } else if (auto operatorExpression = dynamic_cast<OperatorExpression *>(&curAbstractNode)) {
      std::vector<std::unique_ptr<TreeNode>> children(operatorExpression->countChildren());
      //TODO: If not a power-of-two, fill the rest of the children with the neutral element for the operator
      treeNode = std::make_unique<ComputationNode>(
          nullptr,
          std::move(children),
          ComputationOperator(operatorExpression->getOperator(), true));
    } else if (auto variable = dynamic_cast<Variable *>(&curAbstractNode)) {
      //TODO: Handle batchingConstraints if they exist
      std::vector<std::unique_ptr<TreeNode>> children;
      treeNode = std::make_unique<ValueNode>(nullptr, std::move(children), *variable);
    } else if (auto indexAccess = dynamic_cast<IndexAccess *>(&curAbstractNode)) {
      //TODO: Handle batchingConstraints
      std::vector<std::unique_ptr<TreeNode>> children;
      treeNode = std::make_unique<ValueNode>(nullptr, std::move(children), *indexAccess);
    } else if (isLiteral(curAbstractNode)) {
      std::vector<std::unique_ptr<TreeNode>> children;
      treeNode = std::make_unique<ValueNode>(nullptr, std::move(children), curAbstractNode);
    } else if (auto expressionList = dynamic_cast<ExpressionList *>(&curAbstractNode)) {
      //TODO: Implement ExpressionList batching
      throw std::runtime_error("ExpressionLists currently not supported in expression batching.");
    } else if (auto unaryExpression = dynamic_cast<UnaryExpression *>(&curAbstractNode)) {
      //TODO: Implement UnaryExpression batching
      throw std::runtime_error("Unary Expressions currently not supported in expression batching.");
    } else {
      throw std::runtime_error("Unsupported type of AST node: " + std::string(typeid(curAbstractNode).name()));
    }

    // Add the generated TreeNode into the first free slot in the ComputationTree
    if (!computationTree) {
      // First node to be processed becomes the root node of the CT
      computationTree = std::move(treeNode);
      curTreeNode = computationTree.get();
    } else {
      curTreeNode = insertNodeIntoTree(std::move(treeNode), curTreeNode);
    }

    // enqueue children of curNode to process next
    nextLevelNodes.insert(nextLevelNodes.end(), curAbstractNode.begin(), curAbstractNode.end());

    // Are we at the end of a level?
    if (currentLevelNodes.empty()) {
      currentLevelNodes = std::move(nextLevelNodes);
      nextLevelNodes = {};
    }

  } // END WHILE

  return std::move(computationTree);


  /// OLD CODE:

//  // Determine Number of children?
//  if (numChildrenPerNode==-1) {
//    numChildrenPerNode = (int) curNode.countChildren();
//  } else if (numChildrenPerNode!=curNode.countChildren()) {
//    // a subtree is batching incompatible if the children have a different number of operands
//    //TODO: Implement non-symmetric batching
//    throw std::runtime_error("Batching of expressions with different number of children is not yet supported.");
//  }
//
//  if (unprocessedNodes.empty()) {
//
//    // Check that all children are batching compatible to each other
//    AbstractNode &baseNode = childNodes.front().get();
//    for (auto nodeIt = std::next(childNodes.begin()); nodeIt!=childNodes.end(); ++nodeIt) {
//      // check batching compatibility
//      // TODO: Change this to actually construct the new expression as we go!
//      if (!isBatchingCompatible(baseNode, *nodeIt)) {
//        // if we detected a batching incompatibility, we can stop any further testing
//        // TODO: Implement Support for this case!
//        throw std::runtime_error("Rewriting of batching-incompatible expressions is not yet supported.");
//      } else if (isTransparentNode(*nodeIt)) {
//        // as we allow max. 1 transparent node per level, we need to make sure to compare any further transparent
//        // nodes with the one we found here
//        baseNode = *nodeIt;
//      }
//    } // end: for
//
//    // move elements from childNodes to unprocessedNodes: childNodes is empty afterwards
//    unprocessedNodes = std::move(childNodes);
//
//    // reset #children counter back to default value
//    numChildrenPerNode = -1;
//  }

  // if we processed all nodes and did not abort in between due to failed batching compatibility, the node rooted
  // at subtreeRoot is considered as batchable
  //bool isBatchable = qReading.empty() && qWriting.empty();


}

void SpecialExpressionBatcher::visit(AbstractStatement &) {
  throw std::runtime_error("Cannot use the Expression Batcher on statements.");
}

/// Iterate through a computation tree and generate the required code (+batching constraints)?
std::unique_ptr<AbstractNode> SpecialExpressionBatcher::computationTreeToAst(std::unique_ptr<TreeNode> &&computationTree) {

  /// Current index for the __input?__ variables
  unsigned int counter = 0;

  unsigned int depth = 0;

  /// Holds nodes of the current level
  std::deque<std::reference_wrapper<TreeNode>> currentLevelNodes({*computationTree});

  /// Holds nodes of the next level
  std::deque<std::reference_wrapper<TreeNode>> nextLevelNodes;

  std::unique_ptr<Block> block = std::make_unique<Block>();

  std::stack<std::string> resultStack;

  /// Each iteration of the loop corresponds to handling one level of the expression tree
  while (!currentLevelNodes.empty()) {

    // Get the next node to process
    auto &curTreeNode = currentLevelNodes.front().get();
    currentLevelNodes.pop_front();

    // Process the TreeNode node, generating an ast node for it
    if (auto valueNode = dynamic_cast<ValueNode *>(&curTreeNode)) {
//      std::vector<std::unique_ptr<TreeNode>> children;
//      children.emplace_back(nullptr);
//      children.emplace_back(nullptr);
//      astNode = std::make_unique<ComputationNode>(
//          nullptr,
//          std::move(children),
//          ComputationOperator(binaryExpression->getOperator(), true));
//    TODO: HANDLE VALUES
    } else if (auto computationNode = dynamic_cast<ComputationNode *>(&curTreeNode)) {
      if(computationNode->getComputationOperator().isBatched()) {
        // It's internal, so we expect the result to be in a single ciphertext //TODO: How to update this if not at top?
        if(resultStack.empty()) {
          resultStack.push("__input" + std::to_string(counter++) + "__");
        }
        // Rotate it appropriately, depending on depth in computation tree
        auto var = std::make_unique<Variable>(resultStack.top());
        std::vector<std::unique_ptr<AbstractExpression>> v;
        v.emplace_back(std::move(var));
        auto rotation = std::make_unique<Call>("rotate", std::move(v));
        // Now apply the actual operation:
        auto lhs = std::make_unique<Variable>(resultStack.top());
        auto binaryExpression = std::make_unique<BinaryExpression>(std::move(lhs),computationNode->getComputationOperator().getOp(), std::move(rotation));
        // And store it back into the same ciphertext? //TODO: Is this always correct?
        auto target = std::make_unique<Variable>(resultStack.top());
        auto assignment = std::make_unique<Assignment>(std::move(target),std::move(binaryExpression));
        block->prependStatement(std::move(assignment));
      } else {
        //TODO: Handle non-internal operations
      }
    } else {
      throw std::runtime_error("Unsupported type of Tree node: " + std::string(typeid(curTreeNode).name()));
    }



    // enqueue children of curNode to process next
    //TODO: CONVERT BETWEEN UNIQUE PTRS AND REFERENCE WRAPPERS?
    //nextLevelNodes.insert(nextLevelNodes.end(), curTreeNode.getChildren().begin(), curTreeNode.getChildren().end());

    // Are we at the end of a level?
    if (currentLevelNodes.empty()) {
      currentLevelNodes = std::move(nextLevelNodes);
      nextLevelNodes = {};
      depth++;
    }

  } // END WHILE

  return std::unique_ptr<AbstractNode>();
}