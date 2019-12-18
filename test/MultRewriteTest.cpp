#include <gtest/gtest.h>
#include <Ast.h>
#include <include/visitor/MultRewriteVisitor.h>
#include <include/visitor/PrintVisitor.h>
#include <Operator.h>
#include "AstTestingGenerator.h"
#include "Function.h"
#include "BinaryExpr.h"

/// Trivial case where multiplication happens in subsequent statements.
/// Expected: Rewriting
TEST(MultRewriteTest, rewriteSuccessfulSimpleAst) { /* NOLINT */
    Ast ast;
    AstTestingGenerator::getRewritingAst(2, ast);

    // perform rewriting
    MultRewriteVisitor mrv;
    mrv.visit(ast);

    // check for expected changes
    auto func = dynamic_cast<Function *>(ast.getRootNode());
    auto prodDecl = dynamic_cast<VarDecl *>(func->getBody().at(0));
    auto expectedProdDecl = new VarDecl("prod", "int",
                                        new BinaryExpr(
                                                new Variable("inputC"),
                                                OpSymb::multiplication,
                                                new Variable("inputB")));
    // TODO implement (virtual) operator== for AbstractStatement to enable EXPECT_EQ for AbstractStatements
    // TODO implement comparison in derived classes
    // EXPECT_EQ(*prodDecl, *expectedProdDecl);

}

/// Expected: No rewriting
//TEST(MultRewriteTest, rewriteNotApplicable) { /* NOLINT */
//    Ast ast;
//    // generate an AST to be used for rewriting test
//    AstTestingGenerator::getRewritingAst(2, ast);
//
//    std::cout << "test";
//
//
//}