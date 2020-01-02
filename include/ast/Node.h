#ifndef MASTER_THESIS_CODE_NODE_H
#define MASTER_THESIS_CODE_NODE_H

#include <string>

class Node {
 private:
  static int nodeIdCounter;
  std::string uniqueNodeId;

  std::string genUniqueNodeId();

  static int getAndIncrementNodeId();

  static int getNodeIdCounter();

 public:
  [[nodiscard]] virtual std::string getNodeName() const;

  std::string getUniqueNodeId();

  static void resetNodeIdCounter();
};

#endif //MASTER_THESIS_CODE_NODE_H
