#ifdef HAVE_SEAL_BFV

#include "RobertsCross.h"

/// Encrypted RobertsCross, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedRobertsCross(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights)
{
  auto keygenTimer = timer.startTimer();

  // Input Check (output disabled so that benchmark log isn't spammed full of these for every iteration)
  // if (img.size()!=poly_modulus_degree/2) {
  //   std::cout << "WARNING: RobertsCross might be incorrect when img.size() does not match N/2." <<
  //   std::endl;
  // }

  /// Rotations for 3x3 Kernel
  /// Offsets correspond to the different kernel positions
  int img_size = (int)std::sqrt(img.size());
  std::vector<int> rotations = { -img_size + 1, 1,  img_size + 1, -img_size, 0, img_size,
                                 -img_size - 1, -1, img_size - 1 };
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
  keygen.create_galois_keys(rotations, galoisKeys);
  seal::RelinKeys relinKeys;
  keygen.create_relin_keys(relinKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);

  // Create Weight Matrices
  std::vector<int> weight_matrix1 = { 1, 0, 0, 0, -1, 0, 0, 0, 0 };
  std::vector<int> weight_matrix2 = { 0, 1, 0, -1, 0, 0, 0, 0, 0 };
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt the image
  auto encTimer = timer.startTimer();
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  encoder.encode(std::vector<uint64_t>(img.begin(), img.end()), img_ptxt);
  encryptor.encrypt(img_ptxt, img_ctxt);

  // Encode & (if needed) Encrypt the weights
  std::vector<seal::Plaintext> w1_ptxts(weight_matrix1.size()), w2_ptxts(weight_matrix2.size());
  std::vector<seal::Ciphertext> w1_ctxts(weight_matrix1.size()), w2_ctxts(weight_matrix2.size());
  for (size_t i = 0; i < weight_matrix1.size(); ++i)
  {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix1[i]), w1_ptxts[i]);
    if (encrypt_weights)
    {
      encryptor.encrypt(w1_ptxts[i], w1_ctxts[i]);
    }
  }

  for (size_t i = 0; i < weight_matrix2.size(); ++i)
  {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix2[i]), w2_ptxts[i]);
    if (encrypt_weights)
    {
      encryptor.encrypt(w2_ptxts[i], w2_ctxts[i]);
    }
  }
  timer.stopTimer(encTimer);

  auto compTimer = timer.startTimer();
  // FIRST KERNEL:

  // Create rotated copies of the image and multiply by weights
  std::vector<seal::Ciphertext> rotated_img_ctxts(9, seal::Ciphertext(context));
  for (size_t i = 0; i < rotations.size(); ++i)
  {
    evaluator.rotate_rows(img_ctxt, rotations[i], galoisKeys, rotated_img_ctxts[i]);

    if (encrypt_weights)
    {
      evaluator.multiply_inplace(rotated_img_ctxts[i], w1_ctxts[i]);
      evaluator.relinearize_inplace(rotated_img_ctxts[i], relinKeys);
    }
    else
    {
      // If the weight is ptxt and one, we can skip this entirely
      if (weight_matrix1[i] != 1)
      {
        evaluator.multiply_plain_inplace(rotated_img_ctxts[i], w1_ptxts[i]);
      }
    }
  }

  // Sum up all the ciphertexts
  seal::Ciphertext first_result_ctxt(context);
  evaluator.add_many(rotated_img_ctxts, first_result_ctxt);

  // SECOND KERNEL:

  // Create rotated copies of the intermediate result and multiply by weights
  for (size_t i = 0; i < rotations.size(); ++i)
  {
    evaluator.rotate_rows(img_ctxt, rotations[i], galoisKeys, rotated_img_ctxts[i]);

    if (encrypt_weights)
    {
      evaluator.multiply_inplace(rotated_img_ctxts[i], w2_ctxts[i]);
      // relinearization not needed since no more mults coming up
    }
    else
    {
      // If the weight is ptxt and one, we can skip this entirely
      if (weight_matrix2[i] != 1)
      {
        evaluator.multiply_plain_inplace(rotated_img_ctxts[i], w2_ptxts[i]);
      }
    }
  }

  // Sum up all the ciphertexts
  seal::Ciphertext second_result_ctxt(context);
  evaluator.add_many(rotated_img_ctxts, second_result_ctxt);

  // Compute the squares over each result, then add
  evaluator.square_inplace(first_result_ctxt);
  evaluator.square_inplace(second_result_ctxt);
  seal::Ciphertext result_ctxt;
  evaluator.add(first_result_ctxt, second_result_ctxt, result_ctxt);
  timer.stopTimer(compTimer);

  // Decrypt & Return result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);
  return result;
}

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
  auto keygenTimer = timer.startTimer();
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
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt the image
  auto encTimer = timer.startTimer();
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  encoder.encode(std::vector<uint64_t>(img.begin(), img.end()), img_ptxt);
  encryptor.encrypt(img_ptxt, img_ctxt);
  timer.stopTimer(encTimer);

  // Compute
  auto compTimer = timer.startTimer();
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
  timer.stopTimer(compTimer);

  // Decrypt & Return result
  auto decTimer = timer.startTimer();
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(decTimer);
  return result;
}

#endif
