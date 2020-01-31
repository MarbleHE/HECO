#ifndef MASTER_THESIS_CODE_AST_H
#define MASTER_THESIS_CODE_AST_H

#include <map>
#include <string>
#include "../include/ast/Node.h"
#include "../visitor/Visitor.h"

class Ast {
private:
    Node *rootNode;

    std::map<std::string, Literal *> variablesValues;

    bool reversedEdges{false};

public:
    Ast();

    explicit Ast(Node *rootNode);

    ~Ast();

    Node *setRootNode(Node *node);

    [[nodiscard]] Node *getRootNode() const;

    virtual void accept(Visitor &v);

    bool hasVarValue(Variable *var);

    Literal *getVarValue(const std::string &variableIdentifier);

    void updateVarValue(const std::string &variableIdentifier, Literal *newValue);

    Literal *evaluate(std::map<std::string, Literal *> &paramValues, bool printResult);

    void toggleIsReversed();

    [[nodiscard]] bool isReversed() const;

    void printGraphviz();
};

#endif //MASTER_THESIS_CODE_AST_H



