#include <Ast.h>
#include <include/visitor/PrintVisitor.h>
#include "gtest/gtest.h"
#include "examples/genAstDemo.cpp"


TEST(PrintVisitorTest, printDemoTreeTwo) {
    const char *expected =
            "Function:\t[global]\n"
            "\tFunctionParameter: (int)\t[Function_0]\n"
            "\t\tVariable: (encryptedA)\n"
            "\tFunctionParameter: (int)\n"
            "\t\tVariable: (encryptedB)\n"
            "\tVarDecl: (int randInt)\n"
            "\t\tBinaryExpr:\n"
            "\t\t\tCallExternal: (std::rand)\n"
            "\t\t\tOperator: (mod)\n"
            "\t\t\tLiteralInt: (42)\n"
            "\tVarDecl: (bool b)\n"
            "\t\tLogicalExpr:\n"
            "\t\t\tVariable: (encryptedA)\n"
            "\t\t\tOperator: (<)\n"
            "\t\t\tLiteralInt: (2)\n"
            "\tVarDecl: (int sum)\n"
            "\t\tLiteralInt: (0)\n"
            "\tWhile:\n"
            "\t\tLogicalExpr:\n"
            "\t\t\tLogicalExpr:\n"
            "\t\t\t\tVariable: (randInt)\n"
            "\t\t\t\tOperator: (>)\n"
            "\t\t\t\tLiteralInt: (0)\n"
            "\t\t\tOperator: (AND)\n"
            "\t\t\tLogicalExpr:\n"
            "\t\t\t\tUnaryExpr:\n"
            "\t\t\t\t\tOperator: (!)\n"
            "\t\t\t\t\tVariable: (b)\n"
            "\t\t\t\tOperator: (!=)\n"
            "\t\t\t\tLiteralBool: (true)\n"
            "\t\tBlock:\n"
            "\t\t\tVarAssignm: (sum)\t[Block_1]\n"
            "\t\t\t\tBinaryExpr:\n"
            "\t\t\t\t\tVariable: (sum)\n"
            "\t\t\t\t\tOperator: (add)\n"
            "\t\t\t\t\tVariable: (encryptedB)\n"
            "\t\t\tVarAssignm: (randInt)\n"
            "\t\t\t\tBinaryExpr:\n"
            "\t\t\t\t\tVariable: (randInt)\n"
            "\t\t\t\t\tOperator: (sub)\n"
            "\t\t\t\t\tLiteralInt: (1)\n"
            "\tVarDecl: (string outStr)\t[Function_0]\n"
            "\t\tLiteralString: (Computation finished!)\n"
            "\tCallExternal: (printf)\n"
            "\t\tFunctionParameter: (string)\n"
            "\t\t\tVariable: (outStr)\n"
            "\tReturn:\n"
            "\t\tVariable: (sum)\n";
    Ast a;
    generateDemoTwo(a);
    PrintVisitor pv(false);
    pv.visit(a);
    EXPECT_EQ(pv.getOutput(), expected);
}

TEST(PrintVisitorTest, printDemoTreeOne) {
    const char *expected =
            "Function:\t[global]\n"
            "\tFunctionParameter: (int)\t[Function_0]\n"
            "\t\tVariable: (x)\n"
            "\tVarDecl: (int a)\n"
            "\t\tLiteralInt: (4)\n"
            "\tVarDecl: (int k)\n"
            "\tIf:\n"
            "\t\tLogicalExpr:\n"
            "\t\t\tLiteralString: (x)\n"
            "\t\t\tOperator: (>)\n"
            "\t\t\tLiteralInt: (32)\n"
            "\t\tBlock:\n"
            "\t\t\tVarAssignm: (k)\t[Block_1]\n"
            "\t\t\t\tBinaryExpr:\n"
            "\t\t\t\t\tLiteralString: (x)\n"
            "\t\t\t\t\tOperator: (mult)\n"
            "\t\t\t\t\tLiteralString: (a)\n"
            "\t\tBlock:	[Function_0]\n"
            "\t\t\tVarAssignm: (k)\t[Block_2]\n"
            "\t\t\t\tBinaryExpr:\n"
            "\t\t\t\t\tGroup:\n"
            "\t\t\t\t\t\tBinaryExpr:\n"
            "\t\t\t\t\t\t\tLiteralString: (x)\n"
            "\t\t\t\t\t\t\tOperator: (mult)\n"
            "\t\t\t\t\t\t\tLiteralString: (a)\n"
            "\t\t\t\t\tOperator: (add)\n"
            "\t\t\t\t\tLiteralInt: (42)\n"
            "\tReturn:\t[Function_0]\n"
            "\t\tVariable: (k)\n";
    Ast a;
    generateDemoOne(a);
    PrintVisitor pv(false);
    pv.visit(a);
    EXPECT_EQ(pv.getOutput(), expected);
}