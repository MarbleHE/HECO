#include "LinearPolynomial.h"
#ifdef HAVE_SEAL_BFV

/// For a linear polynomial (from the porcupine paper)
/// Ciphertext linear_reg(Ciphertext a, Ciphertext b,
///                       Ciphertext x, Ciphertext y)
///     Ciphertext c1 = multiply(a, x)
///     c1 = relinearize(c1)
///     Ciphertext c2 = sub(y, c1)
///     return sub(c2, b)
std::vector<int64_t> encryptedLinearPolynomialPorcupine(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();

  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  /// Create keys
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt
  auto encTimer = timer.startTimer();
  seal::Plaintext  a_ptxt, b_ptxt, x_ptxt, y_ptxt;
  seal::Ciphertext a_ctxt, b_ctxt, x_ctxt, y_ctxt;

  encoder.encode(std::vector<int64_t>(begin(a), end(a)), a_ptxt);
  encoder.encode(std::vector<int64_t>(begin(b), end(b)), b_ptxt);
  encoder.encode(std::vector<int64_t>(begin(x), end(x)), x_ptxt);
  encoder.encode(std::vector<int64_t>(begin(y), end(y)), y_ptxt);

  encryptor.encrypt(a_ptxt, a_ctxt);
  encryptor.encrypt(b_ptxt, b_ctxt);
  encryptor.encrypt(x_ptxt, x_ctxt);
  encryptor.encrypt(y_ptxt, y_ctxt);
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
  // Ciphertext c1 = multiply(a, x)
  seal::Ciphertext c1;
  evaluator.multiply(a_ctxt, x_ctxt, c1);
  // c1 = relinearize(c1)
  evaluator.relinearize_inplace(c1, relinKeys);
  // Ciphertext c2 = sub(y, c1)
  seal::Ciphertext c2;
  evaluator.sub(y_ctxt, c1, c2);
  // return sub(c2, b)
  seal::Ciphertext result_ctxt;
  evaluator.sub(c2, b_ctxt, result_ctxt);
  timer.stopTimer(compTimer);

  // Decrypt results
  auto decTimer = timer.startTimer();
  std::vector<int64_t> result(a.size());
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);
  return result;
}

/// For a linear polynomial, HECO version (from the porcupine paper)
std::vector<int64_t> encryptedLinearPolynomialBatched(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();

  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  /// Create keys
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt
  auto encTimer = timer.startTimer();
  seal::Plaintext  a_ptxt, b_ptxt, x_ptxt, y_ptxt;
  seal::Ciphertext a_ctxt, b_ctxt, x_ctxt, y_ctxt;

  encoder.encode(std::vector<int64_t>(begin(a), end(a)), a_ptxt);
  encoder.encode(std::vector<int64_t>(begin(b), end(b)), b_ptxt);
  encoder.encode(std::vector<int64_t>(begin(x), end(x)), x_ptxt);
  encoder.encode(std::vector<int64_t>(begin(y), end(y)), y_ptxt);

  encryptor.encrypt(a_ptxt, a_ctxt);
  encryptor.encrypt(b_ptxt, b_ctxt);
  encryptor.encrypt(x_ptxt, x_ctxt);
  encryptor.encrypt(y_ptxt, y_ctxt);
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
  seal::Ciphertext c1;
  evaluator.multiply(a_ctxt, x_ctxt, c1);
  seal::Ciphertext c2;
  evaluator.sub(y_ctxt, c1, c2);
  seal::Ciphertext result_ctxt;
  evaluator.sub(c2, b_ctxt, result_ctxt);
  timer.stopTimer(compTimer);

  // Decrypt results
  auto decTimer = timer.startTimer();
  std::vector<int64_t> result(a.size());
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);
  return result;
}


/// For a linear polynomial (from the porcupine paper)
std::vector<int> encryptedLinearPolynomialNaive(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();

  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  /// Create keys
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt
  auto encTimer = timer.startTimer();
  std::vector<seal::Ciphertext> a_ctxt(a.size()), b_ctxt(b.size()), x_ctxt(x.size()), y_ctxt(y.size());

  for (int i = 0; i < a.size(); ++i) {
    // a
    seal::Plaintext tmp_ptxt;
    std::vector<int64_t> tmp_vector(1, a[i]);

    encoder.encode(tmp_vector, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, a_ctxt[i]);

    // b
    tmp_vector[0] = b[i];
    encoder.encode(tmp_vector, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, b_ctxt[i]);

    // x
    tmp_vector[0] = x[i];
    encoder.encode(tmp_vector, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, x_ctxt[i]);

    // y
    tmp_vector[0] = y[i];
    encoder.encode(tmp_vector, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, y_ctxt[i]);
  }
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
  for (int i = 0; i < a_ctxt.size(); ++i) {
    evaluator.multiply_inplace(a_ctxt[i], x_ctxt[i]);
    evaluator.sub_inplace(y_ctxt[i], a_ctxt[i]);
    evaluator.sub_inplace(y_ctxt[i], b_ctxt[i]);
  }
  timer.stopTimer(compTimer);

  // Decrypt results
  auto decTimer = timer.startTimer();
  std::vector<int> result(y_ctxt.size());

  for (int i = 0; i < y_ctxt.size(); ++i) {
    seal::Plaintext tmp_ptxt;
    std::vector<int64_t> tmp_vector(1);
    decryptor.decrypt(y_ctxt[i], tmp_ptxt);
    encoder.decode(tmp_ptxt, tmp_vector);
    result[i] = (int) tmp_vector[0];
  }

  timer.stopTimer(decTimer);
  return result;
}

#endif