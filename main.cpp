#include "main.h"
#include "examples/genAstDemo.cpp"
#include "include/visitor/PrintVisitor.h"
#include "include/visitor/MultRewriteVisitor.h"

using namespace std;

void demoPrintinJson(bool prettyPrint = true) {
    Ast ast;
    generateDemoOne(ast);
    if (prettyPrint) {
        std::cout << ast.getRootNode()->toJson().dump(2) << std::endl;
    } else {
        std::cout << *ast.getRootNode() << std::endl;
    }
}

void demoPrintingText() {
    Ast ast;
    generateDemoOne(ast);
    PrintVisitor pv;
    pv.visit(ast);
}

void demoRewriting() {
    // Test for the visitor pattern
    Ast ast;
    generateDemoThree(ast);

    // Print the generated tree
    PrintVisitor pv;
    pv.visit(ast);
    std::cout << "------------------------------------------------" << std::endl;

    // Rewrite the tree
    MultRewriteVisitor mrv;
    mrv.visit(ast);
    std::cout << "------------------------------------------------" << std::endl;

    // Print the modified tree
    pv.visit(ast);
    std::cout << "------------------------------------------------" << std::endl;
}


int main() {
    _DEBUG_RUNNING();


    return 0;
}

