#ifndef MASTER_THESIS_CODE_NODE_H
#define MASTER_THESIS_CODE_NODE_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Visitor.h"

using json = nlohmann::json;

class Literal;

class Ast;

class Node {
private:
    std::vector<Node *> children{};

    std::vector<Node *> parents{};

    static int nodeIdCounter;

    std::string uniqueNodeId;

    /// This attributes is used to link back to the original Node in an overlay circuit.
    Node *underlyingNode;

private:

    std::string genUniqueNodeId();

    static int getAndIncrementNodeId();

    static int getNodeIdCounter();

public:

    Node();

    Node *getUnderlyingNode() const;

    void setUnderlyingNode(Node *uNode);

    [[nodiscard]] virtual std::string getNodeName() const;

    std::string getUniqueNodeId();

    static void resetNodeIdCounter();

    [[nodiscard]] const std::vector<Node *> &getChildren() const;

    void addChild(Node *child);

    [[nodiscard]] const std::vector<Node *> &getParents() const;

    [[nodiscard]] const std::vector<Node *> &getPred() const;

    [[nodiscard]] const std::vector<Node *> &getSucc() const;

    void removeChild(Node *child);

    void removeChildren();

    void removeParent(Node *node);

    void removeParents();

    void addChildren(std::vector<Node *> c);

    static void addParent(Node *parentNode, std::vector<Node *> nodesToAddParentTo);

    void addParent(Node *n);

    void swapChildrenParents();

    virtual Literal *evaluate(Ast &ast);

    virtual void accept(Visitor &v);

    [[nodiscard]] virtual json toJson() const;

    [[nodiscard]] virtual std::string toString() const;

    friend std::ostream &operator<<(std::ostream &os, const std::vector<Node *> &v);

    std::string getDotFormattedString(bool isReversed, const std::string &indentation, bool showMultDepth);

    [[nodiscard]] virtual Node *clone();

    void setUniqueNodeId(const std::string &unique_node_id);

    void printAncestorsDescendants(std::vector<Node *> nodes);

    void printAncestorsDescendants(Node *n);
};

#endif //MASTER_THESIS_CODE_NODE_H
