#ifdef HAVE_SEAL_BFV
#include "DotProduct.h"

/// Naive version of dot product, where one ciphertext contains one value.
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
int encryptedDotProductNaive(MultiTimer &timer, std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();
  if (x.size()!=y.size()) throw std::runtime_error("Vectors in dot product must have the same length.");

  // Context Setup
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

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);
  timer.stopTimer(keygenTimer);

  // Encrypt values
  auto encTimer = timer.startTimer();
  std::vector<seal::Ciphertext> x_ctxt(x.size());
  std::vector<seal::Ciphertext> y_ctxt(y.size());

  for (size_t i = 0; i < x.size(); ++i) {
    std::vector<int64_t> elem(1);
    elem[0] = x[i];
    seal::Plaintext tmp;
    encoder.encode(elem, tmp);
    encryptor.encrypt(tmp, x_ctxt[i]);

    elem[0] = y[i];
    encoder.encode(elem, tmp);
    encryptor.encrypt(tmp, y_ctxt[i]);
  }
  seal::Plaintext sum_ptxt;
  seal::Ciphertext sum_ctxt;
  sum_ptxt.set_zero();
  encryptor.encrypt(sum_ptxt, sum_ctxt);
  timer.stopTimer(encTimer);

  // Compute differences
  auto compTimer = timer.startTimer();
  for (size_t i = 0; i < x_ctxt.size(); ++i) {
    // sum += x[i] * y[i]
    evaluator.multiply_inplace(x_ctxt[i], y_ctxt[i]);
    evaluator.add_inplace(sum_ctxt, x_ctxt[i]);
  }
  timer.stopTimer(compTimer);

  auto decTimer = timer.startTimer();
  decryptor.decrypt(sum_ctxt, sum_ptxt);
  std::vector<int64_t> sum_vec(1);
  encoder.decode(sum_ptxt, sum_vec);
  timer.stopTimer(decTimer);

  return (int) sum_vec[0];
}

/// Batched version of dot product, where one ciphertext contains one vector.
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
int encryptedDotProductBatched(MultiTimer &timer, std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();
  size_t vector_size = x.size();
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
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt the vectors
  auto encTimer = timer.startTimer();
  seal::Plaintext x_ptxt, y_ptxt;
  seal::Ciphertext x_ctxt, y_ctxt;
  encoder.encode(std::vector<int64_t>(x.begin(), x.end()), x_ptxt);
  encoder.encode(std::vector<int64_t>(y.begin(), y.end()), y_ptxt);
  encryptor.encrypt(x_ptxt, x_ctxt);
  encryptor.encrypt(y_ptxt, y_ctxt);
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
  evaluator.multiply_inplace(x_ctxt, y_ctxt);
  evaluator.relinearize_inplace(x_ctxt, relinKeys);

  // Fold-and-Sum
  seal::Ciphertext rotation_ctxt;
  for (size_t i = vector_size / 2; i > 0; i /= 2) {
    evaluator.rotate_rows(x_ctxt, i, galoisKeys, rotation_ctxt);
    evaluator.add_inplace(x_ctxt, rotation_ctxt);
  }
  timer.stopTimer(compTimer);

  // Decrypt result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(x_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);

  return (int) result[0];
}

/// Porcupine version of dot product, which is based on the porcupine pseudocode. Only accepts vectors of length 8:
/// Ciphertext dot_product(Ciphertext c0, Plaintext p0)
///     Ciphertext c1 = multiply(c0, p0)
///     Ciphertext c2 = rotate(c1, 2)
///     Ciphertext c3 = add(c1, c2)
///     Ciphertext c4 = rotate(c3, 4)
///     Ciphertext c5 = add(c3, c4)
///     Ciphertext c6 = rotate(c5, 1)
///     return add(c5, c6)
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
int encryptedDotProductPorcupine(MultiTimer &timer, std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree)
{
  // Context Setup
  auto keygenTimer = timer.startTimer();
  if (x.size() != 8 || y.size() != 8) {
    throw std::runtime_error("Vectors in the porcupine version of dot product must have a size 8.");
  }
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
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt the vectors
  auto encTimer = timer.startTimer();
  seal::Plaintext x_ptxt, y_ptxt;
  seal::Ciphertext x_ctxt, y_ctxt;
  encoder.encode(std::vector<int64_t>(x.begin(), x.end()), x_ptxt);
  encoder.encode(std::vector<int64_t>(y.begin(), y.end()), y_ptxt);
  encryptor.encrypt(x_ptxt, x_ctxt);
  encryptor.encrypt(y_ptxt, y_ctxt);
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
  // Ciphertext c1 = multiply(c0, p0)
  seal::Ciphertext c1;
  evaluator.multiply(x_ctxt, y_ctxt, c1);
  evaluator.relinearize_inplace(c1, relinKeys);
  // Ciphertext c2 = rotate(c1, 2)
  seal::Ciphertext c2;
  evaluator.rotate_rows(c1, 2, galoisKeys, c2);
  // Ciphertext c3 = add(c1, c2)
  seal::Ciphertext c3;
  evaluator.add(c1, c2, c3);
  // Ciphertext c4 = rotate(c3, 4)
  seal::Ciphertext c4;
  evaluator.rotate_rows(c3, 4, galoisKeys, c4);
  // Ciphertext c5 = add(c3, c4)
  seal::Ciphertext c5;
  evaluator.add(c3, c4, c5);
  // Ciphertext c6 = rotate(c5, 1)
  seal::Ciphertext c6;
  evaluator.rotate_rows(c5, 1, galoisKeys, c6);
  // return add(c5, c6)
  seal::Ciphertext result_ctxt;
  evaluator.add(c5, c6, result_ctxt);
  timer.stopTimer(compTimer);

  // Decrypt result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);

  return (int) result[0];
}
#endif