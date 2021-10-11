#include "HammingDistance.h"

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
  if (a.size() != 4 || b.size() != 4) {
    std::cout << "WARNING: The porcupine example of hamming distance assumes that 4 elements are given." << std::endl;
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

  // Encode & Encrypt the vectors
  seal::Plaintext a_ptxt, b_ptxt;
  seal::Ciphertext a_ctxt, b_ctxt;
  encoder.encode(std::vector<uint64_t>(a.begin(), a.end()), a_ptxt);
  encoder.encode(std::vector<uint64_t>(b.begin(), b.end()), b_ptxt);
  encryptor.encrypt(a_ptxt, a_ctxt);
  encryptor.encrypt(b_ptxt, b_ctxt);

  // Compute differences
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

  // Decrypt result
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<uint64_t> result;
  encoder.decode(result_ptxt, result);
  return result[0];
}