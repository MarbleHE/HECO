#include <set>
#include <random>
#include "gtest/gtest.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/parser/Parser.h"
#include "ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "ast_opt/visitor/runtime/SealCiphertextFactory.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/visitor/SecretBranchingVisitor.h"
#include "test/ASTComparison.h"

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838
// use a 4x4 matrix for the tests
#define MATRIX_SIZE 4

class KernelTest : public ::testing::Test {  /* NOLINT (predictable sequence expected) */
 protected:
  std::default_random_engine randomEngine;
  std::uniform_int_distribution<int> myUnifIntDist;

  void SetUp() override {
    randomEngine = std::default_random_engine(RAND_SEED);  /* NOLINT (predictable sequence expected) */
    // the supported number range must be according to the FHE scheme parameters to not wrap around the modulo
    myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);
  }

 public:
  void resetRandomEngine() {
    randomEngine.seed(RAND_SEED);
  }

  std::unique_ptr<AbstractNode> getInputs(size_t size) {
    // we use the getInputMatrix method here to create the inputs for the input AST in a reproducible and flexible way
    std::vector<int> inputValues;
    getInputMatrix(size, inputValues);

    std::stringstream inputString;
    inputString << "secret int image = { ";
    std::copy(inputValues.begin(), inputValues.end() - 1, std::ostream_iterator<int>(inputString, ", "));
    inputString << inputValues.back(); // add the last element with no delimiter
    inputString << " };" << std::endl;
    inputString << "int imgSize = " << size <<  ";" << std::endl;

    return Parser::parse(inputString.str());
  }

  static std::unique_ptr<AbstractNode> getOutputs() {
    const char *outputs = R""""(
      resultImage = img2;
    )"""";
    return Parser::parse(std::string(outputs));
  }

  static std::unique_ptr<AbstractNode> getEvaluationProgram() {
    std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
    return getEvaluationProgram(createdNodes);
  }

  static std::unique_ptr<AbstractNode> getEvaluationProgram(std::vector<std::reference_wrapper<AbstractNode>> &createdNodes) {
    // program's input
    const char *inputs = R""""(
      public void runKernel(int imageVec) {
        int weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
        secret int img2 = image;
        for (int x = 1; x < imgSize - 1; x = x + 1) {
          for (int y = 1; y < imgSize - 1; y = y + 1) {
            secret int value = 0;
            for (int j = -1; j < 2; j = j + 1) {
              for (int i = -1; i < 2; i = i + 1) {
                value = value + weightMatrix[i+1][j+1] * image[(x+i)*imgSize+(y+j)];
              }
            }
            img2[x*imgSize+y] = 2 * image[x*imgSize+y] - value;
          }
        }
        return;  // img2 contains result
      }
    )"""";
    return Parser::parse(std::string(inputs), createdNodes);
  }

  void getInputMatrix(size_t size, std::vector<std::vector<int>> &destination) {
    // reset the RNG to make sure that every call to this method results in the same numbers
    resetRandomEngine();
    // make sure we clear desination vector before, otherwise resize could end up appending elements
    destination.clear();
    destination.resize(size, std::vector<int>(size));
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        destination[i][j] = myUnifIntDist(randomEngine);
      }
    }
  }

  void getInputMatrix(size_t size, std::vector<int> &destination) {
    // make sure we clear desination vector before, otherwise resize could end up appending elements
    destination.clear();
    std::vector<std::vector<int>> data;
    getInputMatrix(size, data);
    std::size_t total_size = 0;
    for (const auto &sub : data) total_size += sub.size();
    destination.reserve(total_size);
    for (const auto &sub : data) destination.insert(destination.end(), sub.begin(), sub.end());
  }
};

/// Original, plain C++ program
/// This program is inspired by RAMPARTS, presented by Archer et al. in 2019.
/// The only modification that we do is instead of computing
///     img2[x][y] = img[x][y] - (value/2),
/// we compute
///     img2[x][y] = 2*img[x][y] - value
/// to avoid division that is unsupported in FHE.
std::vector<int> runKernel(std::vector<int> &img) {
  const auto imgSize = (int) ceil(sqrt(img.size()));
  std::vector<std::vector<int>> weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 1; x < imgSize - 1; ++x) {
    for (int y = 1; y < imgSize - 1; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value = value + weightMatrix.at(i + 1).at(j + 1)*img.at((x + i)*imgSize + (y + j));
        }
      }
      img2[imgSize*x + y] = 2*img.at(x*imgSize + y) - value;
    }
  }
  return img2;
}

#if HAVE_SEAL_BFV

/// Check correctness of result between original program and (unmodified) program parsed as AST
TEST_F(KernelTest, originalProgramTest) {  /* NOLINT */
  // run the original program
  std::vector<int> data;
  getInputMatrix(MATRIX_SIZE, data);
  auto expectedResult = runKernel(data);

  // run the unoptimized FHE program
  std::unique_ptr<AbstractNode> evalProgram = KernelTest::getEvaluationProgram();
  std::unique_ptr<AbstractNode> inputs = KernelTest::getInputs(MATRIX_SIZE);
  std::unique_ptr<AbstractNode> outputs = KernelTest::getOutputs();
  auto scf = std::make_unique<SealCiphertextFactory>(16384);

  TypeCheckingVisitor tcv;
  auto rootScope = std::make_unique<Scope>(*evalProgram);

  auto siImage = std::make_unique<ScopedIdentifier>(*rootScope, "image");
  rootScope->addIdentifier("image");
  tcv.addVariableDatatype(*siImage, Datatype(Type::INT, true));

  auto siImgSize = std::make_unique<ScopedIdentifier>(*rootScope, "imgSize");
  rootScope->addIdentifier("imgSize");
  tcv.addVariableDatatype(*siImgSize, Datatype(Type::INT, false));

  tcv.setRootScope(std::move(rootScope));

  evalProgram->accept(tcv);
  auto stn = tcv.getSecretTaintedNodes();

  RuntimeVisitor rtv(*scf, *inputs, tcv.getSecretTaintedNodes());
  // TODO: Implement IndexAccess for secret variables (with plaintext index) and then finish implementing this test

  // we need to use ->begin()->end() as the AST is wrapped into a block (begin()) and then we need to skip the Function
  // statement and instead just visit it's associated Block (end()) because Functions are not supported by the
  // RuntimeVisitor
  // rtv.executeAst(*evalProgram->begin()->end());
  // auto actualResult = rtv.getOutput(*outputs);

  // TODO: Compare expectedResult with actualResult (see RuntimeVisitorTest for examples)
}

#endif

/// Check result generated by TypeCheckingVisitor
TEST_F(KernelTest, STAGE_01_typeCheckingTest) {  /* NOLINT */
  std::vector<int> data;
  KernelTest::getInputMatrix(MATRIX_SIZE, data);
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  std::unique_ptr<AbstractNode> evalProgram = KernelTest::getEvaluationProgram(createdNodes);

  TypeCheckingVisitor tcv;
  auto rootScope = std::make_unique<Scope>(*evalProgram);
  auto siImage = std::make_unique<ScopedIdentifier>(*rootScope, "image");
  rootScope->addIdentifier("image");
  tcv.addVariableDatatype(*siImage, Datatype(Type::INT, true));
  auto siImgSize = std::make_unique<ScopedIdentifier>(*rootScope, "imgSize");
  rootScope->addIdentifier("imgSize");
  tcv.addVariableDatatype(*siImgSize, Datatype(Type::INT, false));
  tcv.setRootScope(std::move(rootScope));
  evalProgram->accept(tcv);

  auto actualTaintedNodes = tcv.getSecretTaintedNodes();

  // code to determine indices of tainted nodes, afterwards check manually if these are really the nodes that should be
  // tainted
//  std::set<std::string> taintedNodesIds;
//  for (auto &[identifier, flag] : actualTaintedNodes) {
//    if (flag) taintedNodesIds.insert(identifier);
//  }
//  for (size_t i = 0; i < createdNodes.size(); ++i) {
//    if (taintedNodesIds.count(createdNodes.at(i).get().getUniqueNodeId()) > 0) std::cout << i << ", " << std::endl;
//  }

  // extract ID dynamically from parsed input program from those nodes that are tainted because node IDs depend on
  // execution order of tests
  std::vector<int> nodesIdxExpectedTainted = {17, 18, 71, 81, 91, 92, 93, 94, 107, 113, 114, 115, 116, 117};
  std::set<std::string> expTainted;
  for (const auto idx : nodesIdxExpectedTainted) {
    expTainted.insert(createdNodes.at(idx).get().getUniqueNodeId());
  }

  for (const auto &[identifier, taintedFlag] : actualTaintedNodes) {
    bool expectedTainted = expTainted.count(identifier) > 0;
    EXPECT_EQ(taintedFlag, expectedTainted);
  }
}

/// Check result after applying CTES
TEST_F(KernelTest, STAGE_02_ctestTest) {  /* NOLINT */
  // Expected: AST is not changed as there are no optimization opportunities without knowing the inputs

  // TODO: Implement this test as soon as CTES has been implemented.
  //  Should be similar as "STAGE_03_secretBranchingRemovalTest" where visited AST is compared with the AST that was
  //  given to the visitor.
}

/// Check result after secret branching removal
TEST_F(KernelTest, STAGE_03_secretBranchingRemovalTest) {  /* NOLINT */
  // Expected: AST is not changed as there are no secret branches to be removed

  std::vector<int> data;
  KernelTest::getInputMatrix(MATRIX_SIZE, data);
  std::unique_ptr<AbstractNode> evalProgram = KernelTest::getEvaluationProgram();

  // get secret tainted nodes map
  TypeCheckingVisitor tcv;
  auto rootScope = std::make_unique<Scope>(*evalProgram);
  auto siImage = std::make_unique<ScopedIdentifier>(*rootScope, "image");
  rootScope->addIdentifier("image");
  tcv.addVariableDatatype(*siImage, Datatype(Type::INT, true));
  auto siImgSize = std::make_unique<ScopedIdentifier>(*rootScope, "imgSize");
  rootScope->addIdentifier("imgSize");
  tcv.addVariableDatatype(*siImgSize, Datatype(Type::INT, false));
  tcv.setRootScope(std::move(rootScope));
  evalProgram->accept(tcv);

  SecretBranchingVisitor sbv(tcv.getSecretTaintedNodes());
  evalProgram->accept(sbv);

  auto expectedOriginalAst = KernelTest::getEvaluationProgram();
  compareAST(*evalProgram, *expectedOriginalAst);
}

/// Check result after loop unrolling
TEST_F(KernelTest, STAGE_04_loopUnrollingTest) {  /* NOLINT */

  // TODO: Implement this test as soon as Loop Unrolling has been implemented.

  // After unrolling inner loop 1
  const char *afterInnerLoop1 = R""""(
      public void runKernel(int imageVec) {
        int weightMatrix = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };
        secret int img2 = image;
        for (int x = 1; x < imgSize - 1; x = x + 1) {
          for (int y = 1; y < imgSize - 1; y = y + 1) {
            int value = 0;
            for (int j = -1; j < 2; j = j + 1) {
              value = value + weightMatrix[0][j+1] * image[(x-1)*imgSize+(y+j)];
              value = value + weightMatrix[1][j+1] * image[(x)*imgSize+(y+j)];
              value = value + weightMatrix[2][j+1] * image[(x+1)*imgSize+(y+j)];
            }
            img2[x][y] = 2*image[x*imgSize+y] - value;
          }
        }
        return;  // img2 contains result
      }
    )"""";
  auto afterInnerLoop1Ast = Parser::parse(std::string(afterInnerLoop1));

  // After unrolling inner loop 2
  const char *afterInnerLoop2 = R""""(
      public void runKernel(int imageVec) {
        int weightMatrix = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };
        secret int img2 = image;
        for (int x = 1; x < imgSize - 1; x = x + 1) {
          for (int y = 1; y < imgSize - 1; y = y + 1) {
            int value = 0;
            value = value + weightMatrix[0][0] * image[(x-1)*imgSize+(y-1)];
            value = value + weightMatrix[1][0] * image[(x)*imgSize+(y-1)];
            value = value + weightMatrix[2][0] * image[(x+1)*imgSize+(y-1)];
            value = value + weightMatrix[0][1] * image[(x-1)*imgSize+(y)];
            value = value + weightMatrix[1][1] * image[(x)*imgSize+(y)];
            value = value + weightMatrix[2][1] * image[(x+1)*imgSize+(y)];
            value = value + weightMatrix[0][2] * image[(x-1)*imgSize+(y+1)];
            value = value + weightMatrix[1][2] * image[(x)*imgSize+(y+1)];
            value = value + weightMatrix[2][2] * image[(x+1)*imgSize+(y+1)];
            img2[x][y] = 2*image[x*imgSize+y] - value;
          }
        }
        return;  // img2 contains result
      }
    )"""";
  auto afterInnerLoop2Ast = Parser::parse(std::string(afterInnerLoop2));

  // After applying CTES on unrolled statements
  const char *afterCtes = R""""(
      public void runKernel(int imageVec) {
        int weightMatrix = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };
        secret int img2 = image;
        for (int x = 1; x < imgSize - 1; x = x + 1) {
          for (int y = 1; y < imgSize - 1; y = y + 1) {
            img2[x][y] = 2*image[x*imgSize+y] -
                (weightMatrix[0][0] * image[(x-1)*imgSize+(y-1)]
                + weightMatrix[1][0] * image[(x)*imgSize+(y-1)]
                + weightMatrix[2][0] * image[(x+1)*imgSize+(y-1)]
                + weightMatrix[0][1] * image[(x-1)*imgSize+(y)]
                + weightMatrix[1][1] * image[(x)*imgSize+(y)]
                + weightMatrix[2][1] * image[(x+1)*imgSize+(y)]
                + weightMatrix[0][2] * image[(x-1)*imgSize+(y+1)]
                + weightMatrix[1][2] * image[(x)*imgSize+(y+1)]
                + weightMatrix[2][2] * image[(x+1)*imgSize+(y+1)]);
          }
        }
        return;  // img2 contains result
      }
    )"""";
  auto afterCtesAst = Parser::parse(std::string(afterCtes));
}

/// Check result after statement vectorization
TEST_F(KernelTest, STAGE_05_statementVectorizationTest) {  /* NOLINT */
  // After applying Vectorizer on unrolled statements
  const char *afterVectorization = R""""(
      public void runKernel(int imageVec) {
        int weightMatrix = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };
        secret int img2 = image;

        // this should ideally be executed using add_many (not implemented in RuntimeVisitor yet)
        __result__ = weightMatrix[0][0] * image
          + rotate(weightMatrix[1][0] * image, imgSize)
          + rotate(weightMatrix[2][0] * image, 2*imgSize)
          + rotate(weightMatrix[0][1] * image, 1)
          + rotate(weightMatrix[1][1] * image, imgSize+1)
          + rotate(weightMatrix[2][1] * image, 2*imgSize+1)
          + rotate(weightMatrix[0][2] * image, 2)
          + rotate(weightMatrix[1][2] * image, imgSize+2)
          + rotate(weightMatrix[2][2] * image, 2*imgSize+2);
        __result_mask__ = { 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        __masked_result__ = __result__ *** __result_mask__;

        // extract border from img2 (actually, the input image as it was copied before)
        img2 = img2 *** ~__result_mask__;

        // merge border-only img2 with rotated masked_result (rotation required to place results in correct slots)
        // Vectorizer should compute:  offset = resultSlotBatchedOutput-targetSlot;
        // For example, offset = 1-6 = -5 => rotate __masked_result__ to the right-hand side by 5
        int offset = -5;
        img2 = img2 + rotate(__masked_result__, offset);

        return;  // img2 contains kernel applied to input image
      }
    )"""";
  auto afterCtesAst = Parser::parse(std::string(afterVectorization));

  // TODO: Implement this test as soon as the Vectorizer has been implemented.
}
