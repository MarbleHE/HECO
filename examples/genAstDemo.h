#ifndef MASTER_THESIS_CODE_GENASTDEMO_H
#define MASTER_THESIS_CODE_GENASTDEMO_H

#include "Ast.h"

/// Provides a REPL-based environment to showcase the different actions that this tool supports.
void runInteractiveDemo();

/// Generates a sample AST for the following code:
///
///  \code{.cpp}
///  int computePrivate(int x) {
///     int a = 4;
///     int k;
///     if (x > 32) {
///         k = x * a;
///     } else {
///         k = (x * a) + 42;
///     }
///     return k;
///  }
/// \endcode
///
/// \param ast The Ast object this tree should be written into.
void generateDemoOne(Ast &ast);

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  int determineSuitableX(int encryptedA, int encryptedB) {
///      int randInt = std::rand() % 42;
///      bool b = encryptedA < 2;
///      int sum = 0;
///
///      while (randInt > 0 && !b == true) {
///          sum = sum + encryptedB;
///          randInt = randInt -1;
///      };
///
///      String outStr = "Computation finished!";
///      printf(outStr);
///
///      return sum;
///  }
///  \endcode
///
/// \param ast The Ast object this tree should be written into.
void generateDemoTwo(Ast &ast);

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  void computeMult() {
///      int a = 3;
///      int b = 7;
///      int c = 9;
///      int result = a * b;
///      result = result * c;
///  }
///  \endcode
///
/// \param ast The Ast object this tree should be written into.
void generateDemoThree(Ast &ast);

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  void computeMult() {
///      int a = 3;
///      int b = 7;
///      int c = 9;
///      int result = a * b;
///      if (4 > 3) {
///         int exampleVal = 3;
///      }
///      result = result * c;
///  }
///  \endcode
///
/// \param ast The Ast object this tree should be written into.
void generateDemoFour(Ast &ast);

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  void computeMult() {
///      int result = inA * (inB * inC);
///      return result;
///  }
///  \endcode
///
///
/// \param ast The Ast object this tree should be written into.
void generateDemoFive(Ast &ast);

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  void multiMult() {
///      int result = (inZ * (inA * (inB * inC)));
///      return result;
///  }
///  \endcode
///
/// \param ast The Ast object this tree should be written into.
void generateDemoSix(Ast &ast);

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  void determineSuitableX(int encryptedA, int encryptedB) {
///      int sum = encryptedA + encryptedB;
///      return sum;
///  }
///  \endcode
///
/// \param ast The Ast object this tree should be written into.
void generateDemoSeven(Ast &ast);

#endif //MASTER_THESIS_CODE_GENASTDEMO_H
