#include "heco/legacy_ast/ast_utilities/PrintVisitor.h"
#include "gtest/gtest.h"
#include "heco/legacy_ast/ast/Assignment.h"
#include "heco/legacy_ast/ast/Literal.h"
#include "heco/legacy_ast/ast/Variable.h"

TEST(PrintVisitor, printTree)
{
    // Confirm that printing children works as expected

    Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

    std::stringstream ss;
    PrintVisitor v(ss);
    v.visit(assignment);

    EXPECT_EQ(
        ss.str(), "NODE VISITED: Assignment\n"
                  "NODE VISITED:   Variable (foo)\n"
                  "LITERAL BOOL VISITED:   LiteralBool (true)\n");
}

// TODO: Extend to non-trivial trees