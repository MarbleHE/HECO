#include "main.h"
#include "examples/genAstDemo.cpp"
#include "include/visitor/PrintVisitor.h"
#include "include/visitor/MultRewriteVisitor.h"

using namespace std;


int main() {
    _DEBUG_RUNNING();

    // Test for the visitor pattern
    Ast ast;
    generateDemoOne(ast);

    // Print the generated tree
    PrintVisitor pv(0);
    pv.visit(ast);

    // Rewrite the tree
//    MultRewriteVisitor mrv();
//    mrv.visit(ast);

    // Print the modified tree
//    pv.resetLevel();
//    pv.visit(ast);

    return 0;
}

