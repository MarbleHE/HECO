#include "LinearKernel.h"

/// Ciphertext linear_reg(Ciphertext a, Ciphertext b, Ciphertext x, Ciphertext y)
///     Ciphertext c1 = multiply(a, x)
///     c1 = relinearize(c1)
///     Ciphertext c2 = sub(y, c1)
///     return sub(c2, b)
int64_t encryptedLinearKernelPorcupine(
        MultiTimer &timer,
        const std::vector<std::vector<float>> &xs,
        const std::vector<float> &ws,
        const std::vector<float> &ys,
        const std::vector<float> &x,
        size_t poly_modulus_degree
)
{
  // Context Setup
  // std::cout << "Setting up SEAL Context" << std::endl;
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  /// Create keys
  // std::cout << "Generating Keys & Helper Objects" << std::endl;
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

  // Encode & Encrypt the inputs
  /// Dimensions per data point
  size_t n = x.size();
  /// Number of existing data points
  size_t m = xs.size();

  seal::Plaintext xs_ptxt;
  seal::Plaintext ws_ptxt;
  seal::Plaintext ys_ptxt;
  seal::Plaintext x_ptxt;
  encoder.encode(std::vector<int64_t>(begin(ws), end(ws)), ws_ptxt);
  encoder.encode(std::vector<int64_t>(begin(ys), end(ys)), ys_ptxt);
  encoder.encode(std::vector<int64_t>(begin(x), end(x)), x_ptxt);

  seal::Ciphertext xs_ctxt;
  seal::Ciphertext ws_ctxt;
  seal::Ciphertext ys_ctxt;
  seal::Ciphertext x_ctxt;
  encryptor.encrypt(ws_ptxt, ws_ctxt);
  encryptor.encrypt(ys_ptxt, ys_ctxt);
  encryptor.encrypt(x_ptxt, x_ctxt);

  // Evaluate
  // Ciphertext c1 = multiply(a, x)
  // c1 = relinearize(c1)
  // Ciphertext c2 = sub(y, c1)
  // return sub(c2, b)
  seal::Ciphertext result_ctxt;

  // Decrypt result
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<uint64_t> result;
  encoder.decode(result_ptxt, result);
  return result[0];
}
