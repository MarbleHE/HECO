#ifdef HAVE_SEAL_BFV

#include "GyKernel.h"

/// Encrypted GxKernel, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedGyKernel(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights)
{

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
std::vector<int64_t> encryptedBatchedGyKernelPorcupine(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree)
{

}

/// Encrypted GxKernel, using vectors of ciphertexts.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int> encryptedNaiveGyKernel(
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
  // First apply [+1  2  +1]
  for (int y = 0; y < img_size; ++y) {
    // Get kernel for first pixel of row y, using padding
    seal::Ciphertext value;
    evaluator.add(img_ctxt.at((y)%img_ctxt.size()), img_ctxt.at((y)%img_ctxt.size()), value);
    evaluator.add_inplace(value, img_ctxt.at((-1*img_size + y)%img_ctxt.size()));
    evaluator.add_inplace(value, img_ctxt.at(img_size + y));
    img2[y] = value;

    // Go through the rest of row y
    for (int x = 1; x < img_size; ++x) {
      // remove the leftmost pixel (old weight +1, now outside kernel)
      //x = middle of current kernel, x-2 = one to the left of kernel
      evaluator.sub_inplace(value, img_ctxt.at(((x - 2)*img_size + y)%img_ctxt.size()));

      // subtract the left pixel (old weight +2, new weight +1)
      // x = middle kernel, x-1 = left element of kernel
      evaluator.sub_inplace(value, img_ctxt.at(((x - 1)*img_size + y)%img_ctxt.size()));

      // add the middle pixel to double it (old weight +1, new weight +2)
      //x = right pixel of previous kernel = middle pixel of new kernel
      evaluator.add_inplace(value, img_ctxt.at(((x)*img_size + y)%img_ctxt.size()));

      // finally, add the right most pixel (no old weight, new weight +1)
      //x = right pixel of previous kernel, x+1 = right pixel of new kernel
      evaluator.add_inplace(value, img_ctxt.at(((x + 1)*img_size + y)%img_ctxt.size()));

      // save result
      img2[x*img_size + y] = value;
    }
  }

  // Now apply the vertical kernel to img2
  // | +1 |
  // |  0 |
  // | -1 |

  // Create new output image
  std::vector<seal::Ciphertext> img3(img2.begin(), img2.end());
  // Vertical Kernel: for each column x
  for (int x = 0; x < img_size; ++x) {
    seal::Ciphertext value;
    // Get kernel for first pixel of column x with padding
    evaluator.sub(img2.at((x*img_size - 1)%img2.size()), img2.at(x*img_size + 1), value);
    // Division that would usually happen here is omitted
    img3[x*img_size + 0] = value;

    // Go through the rest of column x
    for (int y = 1; y < img_size; ++y) {
      // remove the leftmost pixel (old weight +1, now outside kernel)
      //y = middle of current kernel, y-2 = one to the left of kernel
      evaluator.sub_inplace(value, img2.at((x*img_size + y - 2)%img2.size()));

      // add the left pixel (old weight 0, new weight +1)
      // x = middle kernel, x-1 = left element of kernel
      evaluator.add_inplace(value, img2.at((x*img_size + y - 1)%img.size()));

      // add one copy of the middle pixel to cancel out (old weight -1, new weight 0)
      //y = right pixel of previous kernel = middle pixel of new kernel
      evaluator.add_inplace(value, img2.at((x*img_size + y)%img2.size()));

      // finally, subtract the right most pixel (no old weight, new weight +1)
      //y = right pixel of previous kernel, y+1 = right pixel of new kernel
      evaluator.sub_inplace(value, img2.at((x*img_size + y + 1)%img2.size()));

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

#endif