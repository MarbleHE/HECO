#include "QuadraticPolynomial.h"
#ifdef HAVE_SEAL_BFV

///  For a quadratic polynomial (from the porcupine paper)
///  Ciphertext polynomial_reg(Ciphertext a, Ciphertext b,
///                            Ciphertext c, Ciphertext x, Ciphertext y)
///    Ciphertext c1 = multiply(a, x)
///    c1 = relinearize(c1)
///    Ciphertext c2 = add(c1, b)
///    Ciphertext c3 = multiply(x, c2)
///    c3 = relinearize(c3)
///    Ciphertext c4 = add(c3, c)
///    return sub(y, c4)
std::vector<int64_t> encryptedQuadraticPolynomialPorcupine(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> c,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();

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
  seal::Plaintext a_ptxt, b_ptxt, c_ptxt, x_ptxt, y_ptxt;
  seal::Ciphertext a_ctxt, b_ctxt, c_ctxt, x_ctxt, y_ctxt;

  encoder.encode(std::vector<int64_t>(begin(a), end(a)), a_ptxt);
  encoder.encode(std::vector<int64_t>(begin(b), end(b)), b_ptxt);
  encoder.encode(std::vector<int64_t>(begin(c), end(c)), c_ptxt);
  encoder.encode(std::vector<int64_t>(begin(x), end(x)), x_ptxt);
  encoder.encode(std::vector<int64_t>(begin(y), end(y)), y_ptxt);

  encryptor.encrypt(a_ptxt, a_ctxt);
  encryptor.encrypt(b_ptxt, b_ctxt);
  encryptor.encrypt(c_ptxt, c_ctxt);
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
  // Ciphertext c2 = add(c1, b)
  seal::Ciphertext c2;
  evaluator.add(c1, b_ctxt, c2);
  // Ciphertext c3 = multiply(x, c2)
  seal::Ciphertext c3;
  evaluator.multiply(x_ctxt, c2, c3);
  // c3 = relinearize(c3)
  evaluator.relinearize_inplace(c3, relinKeys);
  // Ciphertext c4 = add(c3, c)
  seal::Ciphertext c4;
  evaluator.add(c3, c_ctxt, c4);
  // return sub(y, c4)
  seal::Ciphertext result_ctxt;
  evaluator.sub(y_ctxt, c4, result_ctxt);
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

/// For a quadratic polynomial (from the porcupine paper)
/// This the batched HECO version
std::vector<int64_t> encryptedQuadraticPolynomialBatched(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> c,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();

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
  seal::Plaintext a_ptxt, b_ptxt, c_ptxt, x_ptxt, y_ptxt;
  seal::Ciphertext a_ctxt, b_ctxt, c_ctxt, x_ctxt, y_ctxt;

  encoder.encode(std::vector<int64_t>(begin(a), end(a)), a_ptxt);
  encoder.encode(std::vector<int64_t>(begin(b), end(b)), b_ptxt);
  encoder.encode(std::vector<int64_t>(begin(c), end(c)), c_ptxt);
  encoder.encode(std::vector<int64_t>(begin(x), end(x)), x_ptxt);
  encoder.encode(std::vector<int64_t>(begin(y), end(y)), y_ptxt);

  encryptor.encrypt(a_ptxt, a_ctxt);
  encryptor.encrypt(b_ptxt, b_ctxt);
  encryptor.encrypt(c_ptxt, c_ctxt);
  encryptor.encrypt(x_ptxt, x_ctxt);
  encryptor.encrypt(y_ptxt, y_ctxt);
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();

  evaluator.multiply_inplace(b_ctxt, x_ctxt);
  evaluator.square_inplace(x_ctxt);
  evaluator.relinearize_inplace(x_ctxt, relinKeys);
  evaluator.multiply_inplace(a_ctxt, x_ctxt);

  evaluator.add_inplace(a_ctxt, b_ctxt);
  evaluator.add_inplace(a_ctxt, c_ctxt);

  seal::Ciphertext result_ctxt;
  evaluator.sub(y_ctxt, a_ctxt, result_ctxt);

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

///  For a quadratic polynomial (from the porcupine paper)
/// This is the naive non-batched version
std::vector<int> encryptedQuadraticPolynomialNaive(
        MultiTimer &timer,
        std::vector<int> a,
        std::vector<int> b,
        std::vector<int> c,
        std::vector<int> x,
        std::vector<int> y,
        size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();

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
  std::vector<seal::Ciphertext> a_ctxt(a.size()), b_ctxt(b.size()), c_ctxt(c.size()), x_ctxt(x.size()), y_ctxt(y.size());
  for (int i = 0; i < a.size(); ++i) {
    // a
    seal::Plaintext tmp_ptxt;
    std::vector<int64_t> tmp_vec(1, a[i]);
    encoder.encode(tmp_vec, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, a_ctxt[i]);

    // b
    tmp_vec[0] = b[i];
    encoder.encode(tmp_vec, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, b_ctxt[i]);

    // c
    tmp_vec[0] = c[i];
    encoder.encode(tmp_vec, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, c_ctxt[i]);


    // x
    tmp_vec[0] = x[i];
    encoder.encode(tmp_vec, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, x_ctxt[i]);

    // y
    tmp_vec[0] = y[i];
    encoder.encode(tmp_vec, tmp_ptxt);
    encryptor.encrypt(tmp_ptxt, y_ctxt[i]);
  }
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
  for (int i = 0; i < a_ctxt.size(); ++i) {
    evaluator.multiply_inplace(a_ctxt[i], x_ctxt[i]);
    evaluator.relinearize_inplace(a_ctxt[i], relinKeys);
    evaluator.multiply_inplace(a_ctxt[i], x_ctxt[i]);

    evaluator.multiply_inplace(b_ctxt[i], x_ctxt[i]);

    evaluator.add_inplace(c_ctxt[i], b_ctxt[i]);
    evaluator.add_inplace(c_ctxt[i], a_ctxt[i]);

    evaluator.sub_inplace(y_ctxt[i], c_ctxt[i]);
  }
  timer.stopTimer(compTimer);

  // Decrypt results
  auto decTimer = timer.startTimer();
  std::vector<int> result(y_ctxt.size());
  for (int i = 0; i < y_ctxt.size(); ++i) {
    std::vector<int64_t> tmp_vec(1);
    seal::Plaintext tmp_ptxt;
    decryptor.decrypt(y_ctxt[i], tmp_ptxt);
    encoder.decode(tmp_ptxt, tmp_vec);
    result[i] = (int) tmp_vec[0];
  }
  timer.stopTimer(decTimer);
  return result;
}

#endif