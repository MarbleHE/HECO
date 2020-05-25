#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_EVALUATION_EVALUATIONALGORITHMS_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_EVALUATION_EVALUATIONALGORITHMS_H_

#include <utility>
#include <vector>
#include <ast_opt/ast/Ast.h>
#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#endif

class EvaluationAlgorithms {
 public:

  // ===================================================================================================================
  // ================================================ Plain C++ Versions ===============================================
  // ===================================================================================================================

  /// Implements linear regression: Computes and returns the parameters (a,b) of the regression line y = ax + by.
  /// Implementation based on
  /// https://www.codesansar.com/numerical-methods/linear-regression-method-using-cpp-output.htm.
  /// \param datapoints A vector consisting of (x,y) value pairs.
  /// \return A pair (a,b), the parameters of the regression line y = ax + by.
  static std::pair<float, float> runLinearRegression(std::vector<std::pair<float, float>> datapoints);

  /// Implements the polynomial regression, i.e., computes a,b,c of the polynomial trend line y = a + bx + cx^{2}.
  /// Implementation based on
  /// https://rosettacode.org/wiki/Polynomial_regression#C.2B.2B that in turn implements the formulas described at
  /// https://keisan.casio.com/exec/system/14059932254941.
  /// \param x A vector of the x values.
  /// \param y A vector of the y values associated to the given x values.
  static void runPolynomialRegression(const std::vector<int> &x, const std::vector<int> &y);

  /// Implements the 8-neighbour Laplacian sharpening based on the implementation in the Ramparts paper [1].
  /// The Laplacian sharpening algorithm applies a 3x3-kernel using a stride of 1 on the given image.
  /// References:
  /// [1] Archer, D.W. et al.: RAMPARTS: A Programmer-Friendly System for Building Homomorphic Encryption Applications.
  /// IACR Cryptology ePrint Archive. 2019, 988 (2019).
  /// \param img The 1-channel input image.
  /// \return The resulting image after the Laplacian sharpening has been applied.
  static std::vector<std::vector<int>> runLaplacianSharpeningAlgorithm(std::vector<std::vector<int>> img);

  /// Implements the Sobel edge detection filter based on the implementation in the EVA paper [1] that uses batching.
  /// References:
  /// Dathathri, R. et al.: EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic
  /// Computation. arXiv:1912.11951 [cs]. (2019).
  static std::vector<int> runSobelFilter(const std::vector<int> &img);

  // ===================================================================================================================
  // ==================================================== Plain ASTs ===================================================
  // ===================================================================================================================

  static void genLinearRegressionAst(Ast &ast);

  static void genLaplacianSharpeningAlgorithmAst(Ast &ast);

  static void genPolynomialRegressionAst(Ast &ast);

  static void genSobelFilterAst(Ast &ast);

  // ===================================================================================================================
  // ============================================= ASTs after Running CTES =============================================
  // ===================================================================================================================

  static void genLinearRegressionAstAfterCtes(Ast &ast);

  static void genLaplacianSharpeningAlgorithmAstAfterCtes(Ast &ast);

  static void genPolynomialRegressionAstAfterCtes(Ast &ast);

  static void genSobelFilterAstAfterCtes(Ast &ast);

  // ===================================================================================================================
  // ============================================= Using SEAL directly  ================================================
  // ===================================================================================================================
#ifdef HAVE_SEAL_BFV
  static void encryptedLaplacianSharpeningAlgorithmBatched(std::vector<std::vector<int>> img);
  static void encryptedLaplacianSharpeningAlgorithmNaive(std::vector<std::vector<int>> img);
#endif
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_EVALUATION_EVALUATIONALGORITHMS_H_
