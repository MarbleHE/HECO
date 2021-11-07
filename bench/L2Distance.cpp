#ifdef HAVE_SEAL_BFV

#include "L2Distance.h"

/// Output is squared to elide square root
/// For 4-element distance
/// Ciphertext l2_distance(Ciphertext c0, Ciphertext c1)
///     Ciphertext c2 = sub(c1, c0)
///     Ciphertext c3 = square(c2)
///     c3 = relinearize(c3)
///     Ciphertext c4 = rotate(c3, 2)
///     Ciphertext c5 = add(c3, c4)
///     Ciphertext c6 = rotate(c4, 1)
///     return add(c5, c6)
int encryptedL2DistanceSquared_Porcupine(
        MultiTimer &timer, const std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();
  // TODO: Doesn't work as expected. The summation fails, because the assumption is that the vector is 4 long.
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

  // Compute Euclidean Distance (x[i] - y[i])*(x[i] - y[i]);
  auto compTimer = timer.startTimer();

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
  // Ciphertext c6 = rotate(c4, 1) // TODO: This seems wrong, but I changed it and now it works. Not sure what to do with it
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

/// Output is squared to elide square root
/// Naive version of encrypted l2 distance. Each value will be it's own ciphertext
int encryptedL2DistanceSquared_Naive(
        MultiTimer &timer, const std::vector<int> &x, const std::vector<int> &y, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();
  if (x.size()!=y.size()) throw std::runtime_error("Vectors  in l2 distance must have the same length.");

  // Context Setup
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 20));
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

  for (int i = 0; i < x.size(); ++i) {
    uint64_t elem = x[i];
    seal::Plaintext tmp_a = seal::Plaintext(seal::util::uint_to_hex_string(&elem, std::size_t(1)));
    encryptor.encrypt(tmp_a, x_ctxt[i]);

    elem = y[i];
    seal::Plaintext tmp_b = seal::Plaintext(seal::util::uint_to_hex_string(&elem, std::size_t(1)));
    encryptor.encrypt(tmp_b, y_ctxt[i]);
  }
  seal::Plaintext sum_ptxt;
  seal::Ciphertext sum_ctxt;
  sum_ptxt.set_zero();
  encryptor.encrypt(sum_ptxt, sum_ctxt);
  timer.stopTimer(encTimer);

  // Compute differences
  auto compTimer = timer.startTimer();
  for (size_t i = 0; i < x_ctxt.size(); ++i) {
    // sum += (x[i] - y[i])*(x[i] - y[i]);
    evaluator.sub_inplace(x_ctxt[i], y_ctxt[i]);
    evaluator.square_inplace(x_ctxt[i]);
    evaluator.add_inplace(sum_ctxt, x_ctxt[i]);
  }
  timer.stopTimer(compTimer);

  auto decTimer = timer.startTimer();
  decryptor.decrypt(sum_ctxt, sum_ptxt);
  uint64_t result = *sum_ptxt.data();
  timer.stopTimer(decTimer);

  return (int) result;
}

/// Compute encrypted (squared) L2 distance between two vectors
/// \param a vector of size n
/// \param b vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_both By default, both vectors are encrypted. If set to false, b is plaintext
/// \return L2 distance between the two vectors
int encryptedBatchedSquaredL2Distance(
        MultiTimer &timer, const std::vector<int> &a, const std::vector<int> &b, size_t poly_modulus_degree,
        bool encrypt_both)
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
  seal::Plaintext a_ptxt, b_ptxt;
  seal::Ciphertext a_ctxt, b_ctxt;
  encoder.encode(std::vector<int64_t>(a.begin(), a.end()), a_ptxt);
  encoder.encode(std::vector<int64_t>(b.begin(), b.end()), b_ptxt);
  encryptor.encrypt(a_ptxt, a_ctxt);
  if (encrypt_both)
  {
    encryptor.encrypt(b_ptxt, b_ctxt);
  }
  timer.stopTimer(encTimer);

  // Compute Euclidean Distance (x[i] - y[i])*(x[i] - y[i]);
  auto compTimer = timer.startTimer();
  if (encrypt_both)
  {
    evaluator.sub_inplace(a_ctxt, b_ctxt);
  }
  else
  {
    evaluator.sub_plain_inplace(a_ctxt, b_ptxt);
  }
  evaluator.square_inplace(a_ctxt);
  evaluator.relinearize_inplace(a_ctxt, relinKeys);

  // Fold-and-Sum
  seal::Ciphertext rotation_ctxt;
  // annoyingly, the first rotation has to be done separately
  evaluator.rotate_columns(a_ctxt, galoisKeys, rotation_ctxt);
  evaluator.add_inplace(a_ctxt, rotation_ctxt);
  // Now rotate over the rows automatically
  for (auto i = parameters.poly_modulus_degree() / 4; i > 0; i >>= 1)
  {
    evaluator.rotate_rows(a_ctxt, i, galoisKeys, rotation_ctxt);
    evaluator.add_inplace(a_ctxt, rotation_ctxt);
  }
  timer.stopTimer(compTimer);

  // Decrypt result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(a_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);

  return (int) result[0];
}

#endif
