#ifdef HAVE_SEAL_BFV

#include "HammingDistance.h"

/// \param a vector of size n
/// \param b vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
uint64_t encryptedNaiveHammingDistance(MultiTimer &timer, const std::vector<bool> &a, const std::vector<bool> &b, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();
  if (a.size()!=b.size()) throw std::runtime_error("Vectors  in hamming distance must have the same length.");

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

  // Encrypt values
  auto encTimer = timer.startTimer();
  std::vector<seal::Ciphertext> a_ctxt(b.size());
  std::vector<seal::Ciphertext> b_ctxt(b.size());

  for (int i = 0; i < a.size(); ++i) {
    uint64_t elem = a[i];
    seal::Plaintext tmp_a = seal::Plaintext(seal::util::uint_to_hex_string(&elem, std::size_t(1)));
    encryptor.encrypt(tmp_a, a_ctxt[i]);

    elem = b[i];
    seal::Plaintext tmp_b = seal::Plaintext(seal::util::uint_to_hex_string(&elem, std::size_t(1)));
    encryptor.encrypt(tmp_b, b_ctxt[i]);
  }
  seal::Plaintext sum_ptxt;
  seal::Ciphertext sum_ctxt;
  sum_ptxt.set_zero();
  encryptor.encrypt(sum_ptxt, sum_ctxt);
  timer.stopTimer(encTimer);

  // Compute differences
  auto compTimer = timer.startTimer();
  // Note: We can use the fact that NEQ = XOR = (a-b)^2 for a,b \in {0,1}
  for (int i = 0; i < a_ctxt.size(); ++i) {
    evaluator.sub_inplace(a_ctxt[i], b_ctxt[i]);
    evaluator.square_inplace(a_ctxt[i]);
    evaluator.relinearize_inplace(a_ctxt[i], relinKeys);
    evaluator.add_inplace(sum_ctxt, a_ctxt[i]);
  }
  timer.stopTimer(compTimer);

  auto decTimer = timer.startTimer();
  decryptor.decrypt(sum_ctxt, sum_ptxt);
  uint64_t result = *sum_ptxt.data();
  timer.stopTimer(decTimer);

  return result;
}

/// Computes the encrypted hamming distance between two vectors of booleans
/// Note: Hamming distance over binary vectors can be computed semi-efficiently in Z_p by using NEQ = XOR = (a-b)^2
/// \param a vector of size n
/// \param b vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_both By default, both vectors are encrypted. If set to false, b is plaintext
/// \return
uint64_t encryptedBatchedHammingDistance(
        MultiTimer &timer, const std::vector<bool> &a, const std::vector<bool> &b, size_t poly_modulus_degree, bool encrypt_both)
{
  if (a.size() > poly_modulus_degree / 2)
  {
    std::cerr << "WARNING: HammingDistance might be incorrect when vector size is larger than N/2." << std::endl;
  }

  // Context Setup
  auto keygenTimer = timer.startTimer();
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
  encoder.encode(std::vector<uint64_t>(a.begin(), a.end()), a_ptxt);
  encoder.encode(std::vector<uint64_t>(b.begin(), b.end()), b_ptxt);
  encryptor.encrypt(a_ptxt, a_ctxt);
  if (encrypt_both)
  {
    encryptor.encrypt(b_ptxt, b_ctxt);
  }
  timer.stopTimer(encTimer);

  // Compute differences
  // Note: We can use the fact that NEQ = XOR = (a-b)^2 for a,b \in {0,1}
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
  for (auto i = a.size() / 2; i > 0; i /= 2) {
    evaluator.rotate_rows(a_ctxt, i, galoisKeys, rotation_ctxt);
    evaluator.add_inplace(a_ctxt, rotation_ctxt);
  }
  timer.stopTimer(compTimer);

  // Decrypt result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(a_ctxt, result_ptxt);
  std::vector<uint64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);
  return result[0];
}

/// For 4-element hamming distance
/// Ciphertext hamming_distance(Ciphertext c0, Ciphertext c1)
///     Plaintext p0(N, 2) // N is the number of slots
///     Ciphertext c2 = add(c1, c0)
///     Ciphertext c2_ = negate(c2)
///     Ciphertext c3 = add(c2_, p0)
///     Ciphertext c4 = multiply(c3, c2)
///     c4 = relinearize(c4)
///     Ciphertext c5 = rotate(c4, 2)
///     Ciphertext c6 = add(c4, c5)
///     Ciphertext c7 = rotate(c6, 1)
///     return add(c6, c7)
int encryptedHammingDistancePorcupine(
        MultiTimer &timer, const std::vector<bool> &a, const std::vector<bool> &b, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();
  if (a.size() != 4 || b.size() != 4) {
    throw std::runtime_error("The porcupine example of hamming distance assumes that 4 elements are given.");
  }

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
  encoder.encode(std::vector<uint64_t>(a.begin(), a.end()), a_ptxt);
  encoder.encode(std::vector<uint64_t>(b.begin(), b.end()), b_ptxt);
  encryptor.encrypt(a_ptxt, a_ctxt);
  encryptor.encrypt(b_ptxt, b_ctxt);
  timer.stopTimer(encTimer);

  // Compute differences
  auto compTimer = timer.startTimer();
  // Plaintext p0(N, 2) // N is the number of slots
  std::vector<long> const_vector(encoder.slot_count(), 2);
  seal::Plaintext p0;
  encoder.encode(const_vector, p0);
  // Ciphertext c2 = add(c1, c0)
  seal::Ciphertext c2;
  evaluator.add(a_ctxt, b_ctxt, c2);
  // Ciphertext c2_ = negate(c2)
  seal::Ciphertext c2_;
  evaluator.negate(c2, c2_);
  // Ciphertext c3 = add(c2_, p0)
  seal::Ciphertext c3;
  evaluator.add_plain(c2_, p0, c3);
  // Ciphertext c4 = multiply(c3, c2)
  seal::Ciphertext c4;
  evaluator.multiply(c3, c2, c4);
  // c4 = relinearize(c4)
  evaluator.relinearize(c4, relinKeys, c4);
  // Ciphertext c5 = rotate(c4, 2)
  seal::Ciphertext c5;
  evaluator.rotate_rows(c4, 2, galoisKeys, c5);
  // Ciphertext c6 = add(c4, c5)
  seal::Ciphertext c6;
  evaluator.add(c4, c5, c6);
  // Ciphertext c7 = rotate(c6, 1)
  seal::Ciphertext c7;
  evaluator.rotate_rows(c6, 1, galoisKeys, c7);
  // return add(c6, c7)
  seal::Ciphertext result_ctxt;
  evaluator.add(c6, c7, result_ctxt);
  timer.stopTimer(compTimer);

  // Decrypt result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<uint64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);
  return result[0];
}

#endif