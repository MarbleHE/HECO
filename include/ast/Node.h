#ifndef MASTER_THESIS_CODE_NODE_H
#define MASTER_THESIS_CODE_NODE_H

#include <string>
#include <vector>

class Node {
 private:
  std::vector<Node*> children{};

  std::vector<Node*> parents{};

  static int nodeIdCounter;

  std::string uniqueNodeId;

  std::string genUniqueNodeId();

  static int getAndIncrementNodeId();

  static int getNodeIdCounter();

 public:

  Node();

  [[nodiscard]] virtual std::string getNodeName() const;

  std::string getUniqueNodeId();

  static void resetNodeIdCounter();

  [[nodiscard]] const std::vector<Node*> &getChildren() const;

  void addChild(Node* child);

  [[nodiscard]] const std::vector<Node*> &getParents() const;

  void removeChild(Node* child);

  void removeChildren();

  void removeParent(Node* node);

  void removeParents();

  void addChildren(std::vector<Node*> c);

  static void addParent(Node* parentNode, std::vector<Node*> nodesToAddParentTo);

  void addParent(Node* n);

  void swapChildrenParents();
};

#endif //MASTER_THESIS_CODE_NODE_H
