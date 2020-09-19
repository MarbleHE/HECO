#ifndef AST_OPTIMIZER_TEST_ASTCOMPARISON_H_
#define AST_OPTIMIZER_TEST_ASTCOMPARISON_H_
#include "gtest/gtest.h"

class AbstractNode;

::testing::AssertionResult compareAST(const AbstractNode &ast1, const AbstractNode &ast2);

#endif //AST_OPTIMIZER_TEST_ASTCOMPARISON_H_
