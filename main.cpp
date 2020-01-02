#include "main.h"
#include "examples/genAstDemo.cpp"
#include "include/visitor/PrintVisitor.h"
#include "include/visitor/MultRewriteVisitor.h"

using namespace std;

int main() {
    //_DEBUG_RUNNING();

    //runInteractiveDemo();
    Ast a;
    generateDemoFive(a);

    std::cout << "(1) PrintVisitor ..." << std::endl;
    PrintVisitor pv;
    pv.visit(a);

    std::cout << "(2) MultRewriteVisitor ..." << std::endl;
    MultRewriteVisitor mrv;
    mrv.visit(a);

    std::cout << "(3) PrintVisitor ..." << std::endl;
    pv.visit(a);

    return 0;
}

