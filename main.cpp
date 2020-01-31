#include "main.h"


void buildMultDepthRewritingCircuitSample1(Ast &ast) {
    Function *f = new Function("demoCkt");
    f->addStatement(new Return(new LogicalExpr(
            new LogicalExpr(
                    new LogicalExpr(
                            new LogicalExpr(
                                    new LogicalExpr(
                                            new Variable("a_1^(1)_left"),
                                            OpSymb::logicalAnd,
                                            new Variable("a_1^(1)_right")),
                                    OpSymb::logicalAnd,
                                    new LogicalExpr(
                                            new Variable("a_2^(1)_left"),
                                            OpSymb::logicalXor,
                                            new Variable("a_2^(1)_right"))),
                            OpSymb::logicalXor,
                            new LogicalExpr(
                                    new LogicalExpr(
                                            new Variable("a_1^(2)_left"),
                                            OpSymb::logicalAnd,
                                            new Variable("a_1^(2)_right")),
                                    OpSymb::logicalAnd,
                                    new LogicalExpr(
                                            new Variable("a_2^(2)_left"),
                                            OpSymb::logicalXor,
                                            new Variable("a_2^(2)_right")))),
                    OpSymb::logicalXor,
                    new Variable("y_1")),
            OpSymb::logicalAnd,
            new Variable("a_t"))));
    ast.setRootNode(f);
}

void buildMultDepthRewritingCircuitSample0(Ast &ast) {
    Function *f = new Function("demoCkt");
    f->addStatement(new Return(new LogicalExpr(
            new LogicalExpr(
                    new LogicalExpr(
                            new LogicalExpr(
                                    new Variable("a_1^(1)"),
                                    OpSymb::logicalAnd,
                                    new Variable("a_2^(1)")),
                            OpSymb::logicalXor,
                            new LogicalExpr(
                                    new Variable("a_1^(2)"),
                                    OpSymb::logicalAnd,
                                    new Variable("a_2^(2)"))),
                    OpSymb::logicalXor,
                    new Variable("y_1")),
            OpSymb::logicalAnd,
            new Variable("a_t"))));
    ast.setRootNode(f);
}


int main() {
    //runInteractiveDemo();

    // build demo circuit
    Ast ast;

    buildMultDepthRewritingCircuitSample1(ast);


    // apply cone rewriting to AST
    ConeRewriter::applyConeRewriting(ast);

    return 0;
}


