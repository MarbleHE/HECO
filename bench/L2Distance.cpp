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

