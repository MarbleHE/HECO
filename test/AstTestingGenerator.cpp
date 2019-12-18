#include <iostream>
#include <Operator.h>
#include "AstTestingGenerator.h"
#include "BinaryExpr.h"
#include "VarAssignm.h"
#include "Return.h"
#include "Function.h"

void AstTestingGenerator::getRewritingAst(int id, Ast &ast) {
    switch (id) {
        case 1:
            genAstRewritingOne(ast);
            break;
        case 2:
            genAstRewritingTwo(ast);
            break;
        default:
            std::cerr << "Invalid id given!" << std::endl;
    }
}

void AstTestingGenerator::genAstRewritingOne(Ast &ast) {
    // int computePrivate(int inputA, int inputB, int inputC)
    auto func = new Function("computePrivate");
    auto funcParams = new std::vector<FunctionParameter>();
    funcParams->emplace_back("int", new Variable("inputA"));
    funcParams->emplace_back("int", new Variable("inputB"));
    funcParams->emplace_back("int", new Variable("inputC"));
    func->setParams(funcParams);

    // FIXME this doesn't work yet by MultRewriteVisitor!
    // int prod = [inputA * [inputB * inputC]]
    func->addStatement(new VarDecl("prod", "int",
                                   new BinaryExpr(
                                           new Variable("inputA"),
                                           OpSymb::multiplication,
                                           new BinaryExpr(
                                                   new Variable("inputB"),
                                                   OpSymb::multiplication,
                                                   new Variable("inputC")))));

    // return prod / 3;
    func->addStatement(new Return(new BinaryExpr(
            new Variable("prod"),
            OpSymb::division,
            new LiteralInt(3))));

    ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingTwo(Ast &ast) {
    // int computePrivate(int inputA, int inputB, int inputC)
    auto func = new Function("computePrivate");
    auto funcParams = new std::vector<FunctionParameter>();
    funcParams->emplace_back("int", new Variable("inputA"));
    funcParams->emplace_back("int", new Variable("inputB"));
    funcParams->emplace_back("int", new Variable("inputC"));
    func->setParams(funcParams);

    // int prod = inputA * inputB;
    func->addStatement(new VarDecl("prod", "int",
                                   new BinaryExpr(
                                           new Variable("inputA"),
                                           OpSymb::multiplication,
                                           new Variable("inputB"))));

    // prod = prod * inputC
    func->addStatement(
            new VarAssignm("prod", new BinaryExpr(
                    new Variable("prod"),
                    OpSymb::multiplication,
                    new Variable("inputC"))));

    // return prod / 3;
    func->addStatement(new Return(new BinaryExpr(
            new Variable("prod"),
            OpSymb::division,
            new LiteralInt(3))));

    ast.setRootNode(func);
}
