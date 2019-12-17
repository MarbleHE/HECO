#include "main.h"
#include "examples/genAstDemo.cpp"
#include "include/visitor/PrintVisitor.h"
#include "include/visitor/MultRewriteVisitor.h"

using namespace std;


int main() {
    _DEBUG_RUNNING();

    // Test for the visitor pattern
    Ast ast;
    generateDemoThree(ast);

    // Print the generated tree
    PrintVisitor pv(0);
    pv.visit(ast);
    std::cout << "------------------------------------------------" << std::endl;

    // Rewrite the tree
    MultRewriteVisitor mrv;
    mrv.visit(ast);
    std::cout << "------------------------------------------------" << std::endl;

    // Print the modified tree
    pv.visit(ast);
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}

