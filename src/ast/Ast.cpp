#include "Ast.h"

Ast::Ast(AbstractStatement *rootNode) : rootNode(rootNode) {}

Ast::Ast() {
    rootNode = nullptr;
}

AbstractStatement *Ast::setRootNode(AbstractStatement *rootNode) {
    Ast::rootNode = rootNode;
    return rootNode;
}

AbstractStatement *Ast::getRootNode() const {
    return rootNode;
}

void Ast::accept(Visitor &v) {
    v.visit(*this);
}

Ast::~Ast() {
    delete rootNode;
}
