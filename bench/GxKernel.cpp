#ifdef HAVE_SEAL_BFV

#include "GxKernel.h"

/// Encrypted GxKernel, using vectors of ciphertexts.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int> encryptedNaiveGxKernel(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();
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

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);
  timer.stopTimer(keygenTimer);

  // Encode & Encrypt the image
  auto encTimer = timer.startTimer();
  std::vector<seal::Ciphertext> img_ctxt(img.size());
  for (int i = 0; i < img.size(); ++i) {
    seal::Plaintext tmp;
    encoder.encode(std::vector<int64_t> (1, img[i]), tmp);
    encryptor.encrypt(tmp, img_ctxt[i]);
  }
  timer.stopTimer(encTimer);

  //Compute
  auto compTimer = timer.startTimer();
  std::vector<seal::Ciphertext> img2(img_ctxt.size());

  // First apply [+1  0  -1]
  for (int y = 0; y < img_size; ++y) {
    // Get kernel for first pixel of row y, using padding
    seal::Ciphertext value;
    evaluator.sub(img_ctxt.at((-1*img_size + y)%img_ctxt.size()), img_ctxt.at(1*img_size + y), value);
    img2[0*img_size + y] = value;

    // Go through the rest of row y
    for (int x = 1; x < img_size; ++x) {
      // remove the old leftmost pixel (old weight +1, now outside kernel)
      //x = middle of current kernel, x-2 = one to the left of kernel
      evaluator.sub_inplace(value, img_ctxt.at(((x - 2)*img_size + y)%img.size()));

      // add the left pixel (old weight 0, new weight +1)
      // x = middle kernel, x-1 = left element of kernel
      evaluator.add_inplace(value, img_ctxt.at(((x - 1)*img_size + y)%img.size()));

      // add the middle pixel to zero it out (old weight -1, new weight 0)
      //x = right pixel of previous kernel = middle pixel of new kernel
      evaluator.add_inplace(value, img_ctxt.at(((x)*img_size + y)%img.size()));

      // finally, subtract the right most pixel (no old weight, new weight -1)
      //x = right pixel of previous kernel, x+1 = right pixel of new kernel
      evaluator.sub_inplace(value, img_ctxt.at(((x + 1)*img_size + y)%img.size()));

      // save result
      img2[x*img_size + y] = value;
    }

  }

  // Now apply the vertical kernel to img2
  // | +1 |
  // | +2 |
  // | +1 |

  // Create new output image
  std::vector<seal::Ciphertext> img3(img2.begin(), img2.end());

  // Vertical Kernel: for each column x
  for (int x = 0; x < img_size; ++x) {
    // Get kernel for first pixel of column x with padding
    seal::Ciphertext value;
    seal::Ciphertext doublePixel;
    evaluator.add(img2.at(x*img_size + 0), img2.at(x*img_size + 0), value);
    evaluator.add_inplace(value, img2.at((x*img_size - 1)%img.size()));
    evaluator.add_inplace(value, img2.at(x*img_size + 1));
    // Division that would usually happen here is omitted
    img3[x*img_size + 0] = value;

    // Go through the rest of column x
    for (int y = 1; y < img_size; ++y) {
      // remove the old leftmost pixel (old weight +1, now outside kernel)
      //y = middle of current kernel, y-2 = one to the left of kernel
      evaluator.sub_inplace(value, img2.at((x*img_size + y - 2)%img.size()));

      // subtract the left pixel (old weight +2, new weight +1)
      // x = middle kernel, x-1 = left element of kernel
      evaluator.sub_inplace(value, img2.at((x*img_size + y - 1)%img.size()));

      // add one copy of the middle pixel (old weight +1, new weight +2)
      //y = right pixel of previous kernel = middle pixel of new kernel
      evaluator.add_inplace(value, img2.at((x*img_size + y)%img.size()));

      // finally, add the right most pixel (no old weight, new weight +1)
      //y = right pixel of previous kernel, y+1 = right pixel of new kernel
      evaluator.add_inplace(value, img2.at((x*img_size + y + 1)%img.size()));

      // save result
      img3[x*img_size + y] = value;
    }
  }

  timer.stopTimer(compTimer);

  // Decrypt results
  auto decTimer = timer.startTimer();
  std::vector<int> result(img.size());
  for (int i = 0; i < result.size(); ++i) {
    seal::Plaintext tmp;
    decryptor.decrypt(img3[i], tmp);

    std::vector<int64_t> tmp_vec(1);
    encoder.decode(tmp, tmp_vec);
    result[i] = (int) tmp_vec[0];
  }
  timer.stopTimer(decTimer);
  return result;
}


/// Encrypted GxKernel, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedGxKernel(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights)
{
  auto keygenTimer = timer.startTimer();

  // Input Check
  if (img.size() != poly_modulus_degree / 2)
  {
    std::cerr << "WARNING: BatchedGxKernel might be incorrect when img.size() does not match N/2." << std::endl;
  }

  /// Rotations for 3x3 Kernel
  /// Offsets correspond to the different kernel positions
  // Since the middle weights are zero, we can actually get rid of those rotations
  int img_size = (int)std::sqrt(img.size());
  std::vector<int> rotations = { -img_size + 1,
          // 1,
                                 img_size + 1, -img_size,
          // 0,
                                 img_size, -img_size - 1,
          //-1,
                                 img_size - 1 };

  // Context Setup
  // std::cout << "Setting up SEAL Context" << std::endl;
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 60));
  seal::SEALContext context(parameters);

  /// Create keys
  // std::cout << "Generating Keys & Helper Objects" << std::endl;
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

  timer.stopTimer(keygenTimer);

  auto encTimer = timer.startTimer();
  // Create Weight Matrix.
  ///        | +1  0  -1 |
  ///   w =  | +2  0  -2 |
  ///        | +1  0  -1 |
  // Since the middle weights are zero, we got rid of them already by not rotating those
  std::vector<int> weight_matrix = { 1, -1, +2, -2, 1, -1 };

  // Encode & Encrypt the image
  // std::cout << "Encoding & Encrypting Image" << std::endl;
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  encoder.encode(std::vector<uint64_t>(img.begin(), img.end()), img_ptxt);
  encryptor.encrypt(img_ptxt, img_ctxt);

  // Encode & (if needed) Encrypt the weights
  std::vector<seal::Plaintext> w_ptxts(weight_matrix.size());
  std::vector<seal::Ciphertext> w_ctxts(weight_matrix.size());
  for (size_t i = 0; i < weight_matrix.size(); ++i)
  {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix[i]), w_ptxts[i]);
    if (encrypt_weights)
    {
      encryptor.encrypt(w_ptxts[i], w_ctxts[i]);
    }
  }

  timer.stopTimer(encTimer);

  // Create rotated copies of the image and multiply by weights
  // std::cout << "Applying Kernel" << std::endl;
  auto compTimer = timer.startTimer();
  std::vector<seal::Ciphertext> rotated_img_ctxts(9, seal::Ciphertext(context));
  for (size_t i = 0; i < rotations.size(); ++i)
  {
    evaluator.rotate_rows(img_ctxt, rotations[i], galoisKeys, rotated_img_ctxts[i]);

    if (encrypt_weights)
    {
      evaluator.multiply_inplace(rotated_img_ctxts[i], w_ctxts[i]);
      // relinearization not needed since no more mults coming up
    }
    else
    {
      // If the weight is ptxt and one, we can skip this entirely
      if (weight_matrix[i] != 1)
      {
        evaluator.multiply_plain_inplace(rotated_img_ctxts[i], w_ptxts[i]);
      }
    }
  }

  // Sum up all the ciphertexts
  seal::Ciphertext result_ctxt(context);
  evaluator.add_many(rotated_img_ctxts, result_ctxt);

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

/// Encrypted GxKernel, ported from Porcupine, according to:
/// Ciphertext gx(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, w)
///     Ciphertext c2 = add(c0, c1)
///     Ciphertext c3 = rotate(c2, -w)
///     Ciphertext c4 = add(c2, c3)
///     Ciphertext c5 = rotate(c4, 1)
///     Ciphertext c6 = rotate(c4, -1)
///     return sub(c5, c6)
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int64_t> encryptedBatchedGxKernelPorcupine(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree)
{
  auto keygenTimer = timer.startTimer();

  // Input Check
  if (img.size() != poly_modulus_degree / 2)
  {
    std::cerr << "WARNING: BatchedGxKernel might be incorrect when img.size() does not match N/2." << std::endl;
  }

  int img_size = (int) std::sqrt(img.size());

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

  // Compute new image
  auto compTimer = timer.startTimer();
  // Ciphertext c1 = rotate(c0, w)
  seal::Ciphertext c1;
  evaluator.rotate_rows(img_ctxt, -1, galoisKeys, c1); // img_ctxt == c0
  // Ciphertext c2 = add(c0, c1)
  seal::Ciphertext c2;
  evaluator.add(img_ctxt, c1, c2);
  // evaluator.add(img_ctxt, c1, c1); // c1 == c2
  // Ciphertext c3 = rotate(c2, -w)
  seal::Ciphertext c3;
  evaluator.rotate_rows(c2, 1, galoisKeys, c3);
  // evaluator.rotate_rows(c1, -1 * img_size, galoisKeys, img_ctxt); // img_ctxt == c3
  // Ciphertext c4 = add(c2, c3)
  seal::Ciphertext c4;
  evaluator.add(c2, c3, c4);
  // evaluator.add(c1, img_ctxt, c1); // c1 == c4
  // Ciphertext c5 = rotate(c4, 1)
  seal::Ciphertext c5;
  evaluator.rotate_rows(c4, -1 * img_size, galoisKeys, c5);
  // evaluator.rotate_rows(c1, 1, galoisKeys, img_ctxt); // img_ctxt == c5
  // Ciphertext c6 = rotate(c4, -1)
  seal::Ciphertext c6;
  evaluator.rotate_rows(c4, img_size, galoisKeys, c6);
  // evaluator.rotate_rows_inplace(c1, -1, galoisKeys); // c1 == c6
  // return sub(c5, c6)
  seal::Ciphertext result_ctxt;
  evaluator.sub(c5, c6, result_ctxt);
  // evaluator.sub(img_ctxt, c1, img_ctxt);
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