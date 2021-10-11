#ifdef HAVE_SEAL_BFV

#include "RobertsCross.h"

/// Output is squared to elide square root
/// Ciphertext roberts_cross(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, w)
///     Ciphertext c2 = rotate(c0, 1)
///     Ciphertext c3 = sub(c1, c2)
///     Ciphertext c4 = square(c4)
///     c4 = relinearize(c4)
///     Ciphertext c5 = rotate(c0, w + 1)
///     Ciphertext c6 = sub(c5, c0)
///     Ciphertext c7 = square(c6)
///     c7 = relinearize(c7)
///     return add(c4, c7)
std::vector<int64_t> encryptedRobertsCrossPorcupine(
        MultiTimer &timer, std::vector<int> &img, size_t poly_modulus_degree)
{
  // TODO: Doesn't work as expected
  int img_size = (int)std::sqrt(img.size());

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
  seal::GaloisKeys galoisKeys;
  keygen.create_galois_keys(galoisKeys);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);

  // Encode & Encrypt the image
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  encoder.encode(std::vector<uint64_t>(img.begin(), img.end()), img_ptxt);
  encryptor.encrypt(img_ptxt, img_ctxt);

  // Ciphertext c1 = rotate(c0, w)
  seal::Ciphertext c1;
  evaluator.rotate_rows(img_ctxt, img_size, galoisKeys, c1);
  // Ciphertext c2 = rotate(c0, 1)
  seal::Ciphertext c2;
  evaluator.rotate_rows(img_ctxt, 1, galoisKeys, c2);
  // Ciphertext c3 = sub(c1, c2)
  seal::Ciphertext c3;
  evaluator.sub(c1, c2, c3);
  // Ciphertext c4 = square(c4) //TODO: There is an error in the pseudo-code here
  seal::Ciphertext c4;
  evaluator.square(c3, c4);
  // c4 = relinearize(c4)
  evaluator.relinearize_inplace(c4, relinKeys);
  // Ciphertext c5 = rotate(c0, w + 1)
  seal::Ciphertext c5;
  evaluator.rotate_rows(img_ctxt, img_size + 1, galoisKeys, c5);
  // Ciphertext c6 = sub(c5, c0)
  seal::Ciphertext c6;
  evaluator.sub(c5, img_ctxt, c6);
  // Ciphertext c7 = square(c6)
  seal::Ciphertext c7;
  evaluator.square(c6, c7);
  // c7 = relinearize(c7)
  evaluator.relinearize_inplace(c7, relinKeys);
  // return add(c4, c7)
  seal::Ciphertext result_ctxt;
  evaluator.add(c4, c7, result_ctxt);

  // Decrypt & Return result
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  return result;
}

#endif
