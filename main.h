
#ifndef MASTER_THESIS_CODE_MAIN_H
#define MASTER_THESIS_CODE_MAIN_H

#include <iostream>
#include "include/ast/LiteralInt.h"
#include "include/ast/LiteralString.h"
#include "include/ast/VarDecl.h"
#include "include/ast/Variable.h"
#include "include/ast/Block.h"
#include "include/ast/BinaryExpr.h"
#include "include/ast/VarAssignm.h"
#include "include/ast/Group.h"
#include "include/ast/If.h"
#include "include/ast/Return.h"
#include "include/ast/Function.h"
#include "include/ast/FunctionParameter.h"
#include "include/ast/AbstractStatement.h"
#include "include/ast/While.h"
#include "include/ast/LogicalExpr.h"
#include "include/ast/LiteralBool.h"
#include "include/ast/UnaryExpr.h"
#include "include/ast/CallExternal.h"


/// Program's entry point.
int main();

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  int computePrivate(int x) {        // Function
///     int a = 4;                      // VarDecl, LiteralInt
///     int k;                          // VarDecl
///     if (x > 32) {                   // If, Block, Variable
///         k = x * a;                  // VarAssignm, BinaryExpr, Operator, Variable
///     } else {                        // Block
///         k = (x * a) + 42;           // VarAssignm, Group, BinaryExpr, BinaryExpr, Variable
///     }
///     return k;                       // Return
///  }
/// \endcode
///
Function generateDemoOne();

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  int determineSuitableX(int encryptedA, int encryptedB) {
///      int randInt = rand() % 42;                  // Call
///      bool b = encryptedA < 2;                    // LiteralBool
///      int sum = 0;                                // LiteralInt
///
///      while (randInt > 0 && !b == true) {         // While, LogicalExpr, UnaryExpr
///          sum = sum + encryptedB;                 // VarAssignm, BinaryExpr
///          randInt--;                              // BinaryExpr
///      };
///
///      String outStr = "Computation finished!";    // LiteralString
///      printf(outStr);
///
///      return sum;
///  }
///  \endcode
Function generateDemoTwo();

#endif //MASTER_THESIS_CODE_MAIN_H
