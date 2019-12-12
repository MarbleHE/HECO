
#include "Ast.h"

Ast::Ast(AbstractStatement *rootNode) : rootNode(rootNode) {}

Ast::Ast() {

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
