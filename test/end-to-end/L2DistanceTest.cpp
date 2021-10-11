#include <vector>
#include <stdexcept>
#include "ast_opt/runtime/RuntimeVisitor.h"
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/parser/Parser.h"
#include "gtest/gtest.h"

#include "L2DistanceTest.h"

/// Original, plain C++ program for the (squared) L2 distance between two vectors
///
/// Since sqrt() is not supported in BFV, we omit it and compute the squared distance
/// \param x vector of size n
/// \param y vector of size n
/// \return L2 distance between the two vectors
int squaredL2Distance(const std::vector<int> &x, const std::vector<int> &y) {

  if (x.size()!=y.size()) throw std::runtime_error("Vectors in L2 distance must have the same length.");
  int sum = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sum += (x[i] - y[i])*(x[i] - y[i]);
  }
  return sum;
}

#ifdef HAVE_SEAL_BFV
/// Output is squared to elide square root
/// For 4-element distance
/// Ciphertext l2_distance(Ciphertext c0, Ciphertext c1)
///     Ciphertext c2 = sub(c1, c0)/
///     Ciphertext c3 = square(c2)/
///     c3 = relinearize(c3)/
///     Ciphertext c4 = rotate(c3, 2)/
///     Ciphertext c5 = add(c3, c4)/
///     Ciphertext c6 = rotate(c4, 1)/
///     return add(c5, c6)/
int encryptedL2DistanceSquared_Porcupine(
        MultiTimer &timer, const std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree)
{
  if (x.size() != 4 || y.size() != 4) {
    std::cout << "WARNING: The porcupine example of l2 distance assumes that 4 elements are given." << std::endl;
  }

  // Context Setup
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 60));
  seal::SEALContext context(parameters);

  /// Create keys
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);
  seal::GaloisKeys galoisKeys;
  keygen.create_galois_keys(galoisKeys);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);

  // Encode & Encrypt the vectors
  seal::Plaintext x_ptxt, y_ptxt;
  seal::Ciphertext x_ctxt, y_ctxt;
  encoder.encode(std::vector<int64_t>(x.begin(), x.end()), x_ptxt);
  encoder.encode(std::vector<int64_t>(y.begin(), y.end()), y_ptxt);
  encryptor.encrypt(x_ptxt, x_ctxt);
  encryptor.encrypt(y_ptxt, y_ctxt);

  // Compute Euclidean Distance (x[i] - y[i])*(x[i] - y[i]);
  // Ciphertext c2 = sub(c1, c0)
  seal::Ciphertext c2;
  evaluator.sub(y_ctxt, x_ctxt, c2);
  // Ciphertext c3 = square(c2)
  seal::Ciphertext c3;
  evaluator.square(c2, c3);
  // c3 = relinearize(c3)
  evaluator.relinearize(c3, relinKeys, c3);

  // Ciphertext c4 = rotate(c3, 2)
  seal::Ciphertext c4;
  evaluator.rotate_rows(c3, 2, galoisKeys, c4);
  // Ciphertext c5 = add(c3, c4)
  seal::Ciphertext c5;
  evaluator.add(c3, c4, c5);
  // Ciphertext c6 = rotate(c4, 1)
  seal::Ciphertext c6;
  evaluator.rotate_rows(c4, 1, galoisKeys, c6);
  // return add(c5, c6)
  seal::Ciphertext result_ctxt;
  evaluator.add(c5, c6, result_ctxt);

  // Decrypt result
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);

  for (auto elem : result) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;

  return result[0];
}

TEST(SquaredL2DistanceTest, Clear_EncryptedPorcupine_Equivalence) { /* NOLINT */
  std::vector<int> a(4, 50); // TODO: Create from fixed random seed
  std::vector<int> b(4, 40); // TODO: Create from fixed random seed

  MultiTimer dummy = MultiTimer();
  auto result = encryptedL2DistanceSquared_Porcupine(dummy, a, b, 2 << 13);

  // Compare to reference cleartext implementation
  EXPECT_EQ(squaredL2Distance(a, b), result);
}

#endif //HAVE_SEAL_BFV

TEST(SquaredL2DistanceTest, clearTextEvaluation) { /* NOLINT */
  /// program's input
  const char *inputs = R""""(
        int x = {10,14,17,0};
        int y = {5,14,12,0};
        int n = 4;
      )"""";
  auto astInput = Parser::parse(std::string(inputs));


  /// program specification
  const char *program = R""""(
      int sum = 0;
      for (int i = 0; i < n; i = i + 1) {
        sum = sum + (x[i]-y[i])*(x[i]-y[i]);
      }
      return sum;
      )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
        sum = sum;
      )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  auto scf = std::make_unique<DummyCiphertextFactory>();
  auto tcv = std::make_unique<TypeCheckingVisitor>();

  // create and prepopulate TypeCheckingVisitor
  auto registerInputVariable = [&tcv](Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  };

  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "x", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "y", Datatype(Type::INT, false));
  registerInputVariable(*rootScope, "n", Datatype(Type::INT, false));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  //TODO: Change it so that by passing in an empty secretTaintingMap, we can get the RuntimeVisitor to execute everything "in the clear"!
  auto empty = std::unordered_map<std::string, bool>();
  RuntimeVisitor srv(*scf, *astInput, empty);
  srv.executeAst(*astProgram);


  /// A helper method that takes the result produced by the RuntimeVisitor (result) and a list of expected
  /// (identifier, vector of values) pairs that the program should have returned.
  /// \param result The generated result retrieved by getOutput from the RuntimeVisitor.
  /// \param expectedResult The expected result that the program should have been produced.
  auto assertResult = [&scf](const OutputIdentifierValuePairs &result,
                             const std::unordered_map<std::string, std::vector<int64_t>> &expectedResult) {
    // Check that the number of results match the number of expected results
    EXPECT_EQ(result.size(), expectedResult.size());

    for (const auto &[identifier, cipherClearText] : result) {
      // Check that the result we are currently processing is indeed an expected result
      EXPECT_EQ(expectedResult.count(identifier), 1);

      // for checking the value, distinguish between a ciphertext (requires decryption) and plaintext
      std::vector<int64_t> plainValues;
      if (auto ciphertext = dynamic_cast<AbstractCiphertext *>(cipherClearText.get())) {        // result is a ciphertxt
        scf->decryptCiphertext(*ciphertext, plainValues);
        const auto &expResultVec = expectedResult.at(identifier);
        // to avoid comparing the expanded values (last element of ciphertext is repeated to all remaining slots), we
        // only compare the values provided in the expectedResult map
        for (int i = 0; i < expResultVec.size(); ++i) {
          EXPECT_EQ(plainValues.at(i), expectedResult.at(identifier).at(i));
        }
      } else if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(cipherClearText.get())) {   // result is a cleartext
        auto cleartextData = cleartextInt->getData();
        // required to convert vector<int> to vector<int64_t>
        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
        EXPECT_EQ(plainValues, expectedResult.at(identifier));
      } else if (auto
          cleartextBool = dynamic_cast<Cleartext<bool> *>(cipherClearText.get())) {   // result is a cleartext
        auto cleartextData = cleartextBool->getData();
        // required to convert vector<int> to vector<int64_t>
        plainValues.insert(plainValues.end(), cleartextData.begin(), cleartextData.end());
        EXPECT_EQ(plainValues, expectedResult.at(identifier));
      } else {
        throw std::runtime_error("Could not determine type of result.");
      }
    }
  };

  std::unordered_map<std::string, std::vector<int64_t>> expectedResult;
  expectedResult["sum"] = {50};
  auto result = srv.getOutput(*astOutput);
  assertResult(result, expectedResult);
}