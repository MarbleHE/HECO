#ifdef HAVE_SEAL_BFV

#include "BoxBlur.h"

std::vector<uint64_t> encryptedFastBoxBlur2x2(MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree) {
  auto keygenTimer = timer.startTimer();
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));

  // Context Setup
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_poly_modulus_degree(poly_modulus_degree);
  parameters.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parameters.poly_modulus_degree()));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(parameters.poly_modulus_degree(), 30));
  seal::SEALContext context(parameters);

  // Create keys
  seal::KeyGenerator keygen(context);
  seal::SecretKey secretKey = keygen.secret_key();
  seal::PublicKey publicKey;
  keygen.create_public_key(publicKey);

  // Create helper objects
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);
  timer.stopTimer(keygenTimer);

  auto encryptTimer = timer.startTimer();
  std::vector<seal::Ciphertext> encImg(img.size());
  for (int i = 0; i < img.size(); ++i) {
    // Here we will assume that img only contains positive values in order to reuse seal utilities
    uint64_t pixel = img[i];
    seal::Plaintext ptx = seal::Plaintext(seal::util::uint_to_hex_string(&pixel, std::size_t(1)));
    encryptor.encrypt(ptx, encImg[i]);
  }
  timer.stopTimer(encryptTimer);

  auto compTimer = timer.startTimer();
  std::vector<seal::Ciphertext> img2(encImg.begin(), encImg.end());

  // Horizontal Kernel: for each row y
  for (int y = 0; y < imgSize; ++y) {
    // Get kernel for first pixel of row y, using padding
    seal::Ciphertext value;
    evaluator.add(encImg.at((-1*imgSize + y) % encImg.size()), encImg.at(0*imgSize + y), value);
    // Division that would usually happen here is omitted
    img2[0*imgSize + y] = value; // Is this gonna copy or just have the reference?

    // Go through the rest of row y
    for (int x = 1; x < imgSize; ++x) {
      // remove the previous pixel
      evaluator.sub(value, encImg.at(((x - 2)*imgSize + y) % encImg.size()), value);
      // add the new pixel
      evaluator.add(value, encImg.at((x*imgSize + y) % encImg.size()), value);
      // save result
      img2[x*imgSize + y] = value;
    }
  }

  // Now apply the vertical kernel to img2

  // Create new output image
  std::vector<seal::Ciphertext> img3(img2.begin(), img2.end());

  // Vertical Kernel: for each column x
  for (int x = 0; x < imgSize; ++x) {
    // Get kernel for first pixel of column x with padding
    seal::Ciphertext value;
    evaluator.add(img2.at((x*imgSize - 1) % img.size()), img2.at(x*imgSize + 0), value);
    // Division that would usually happen here is omitted
    img3[x*imgSize + 0] = value;

    // Go through the rest of column x
    for (int y = 1; y < imgSize; ++y) {
      // remove the previous pixel
      evaluator.sub(value, img2.at((x*imgSize + y - 2) % img.size()), value);
      // add the new pixel
      evaluator.add(value, img2.at((x*imgSize + y) % img.size()), value);
      // save result
      img3[x*imgSize + y] = value;
    }
  }
  timer.stopTimer(compTimer);

  auto decryptTimer = timer.startTimer();
  std::vector<uint64_t> result(img.size());
  for (int i = 0; i < img3.size(); ++i) {
    seal::Plaintext ptx;
    decryptor.decrypt(img3[i], ptx);
    result[i] = *ptx.data();
  }
  timer.stopTimer(decryptTimer);

  return result;
}


/// Encrypted BoxBlur, using 3x3 Kernel batched as 9 rotations of the image
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \param encrypt_weights By default, the kernel weights are plaintexts. If this is set, they are also ciphertexts.
/// \return transformed image
std::vector<int64_t> encryptedBatchedBoxBlur(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights)
{
  int t0 = timer.startTimer(); // keygen timer start
  // Input Check
  if (img.size() != poly_modulus_degree / 2)
  {
    std::cerr << "WARNING: BatchedBoxBlur might be incorrect when img.size() does not match N/2." << std::endl;
  }

  /// Rotations for 3x3 Kernel
  /// Offsets correspond to the different kernel positions
  int img_size = (int)std::sqrt(img.size());
  std::vector<int> rotations = { -img_size + 1, 1,  img_size + 1, -img_size, 0, img_size,
                                 -img_size - 1, -1, img_size - 1 };
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
  seal::GaloisKeys galoisKeys;
  keygen.create_galois_keys(rotations, galoisKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context); // changed from this: EVALUATOR evaluator(context);

  // Create Weight Matrix
  std::vector<int> weight_matrix = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  timer.stopTimer(t0); // keygen timer stop

  auto encryptTime = timer.startTimer(); // encryption timer start
  // Encode & Encrypt the image
  // std::cout << "Encoding & Encrypting Image" << std::endl;
  seal::Plaintext img_ptxt;
  seal::Ciphertext img_ctxt;
  std::vector<uint64_t> long_vec = std::vector<uint64_t>(img.begin(), img.end());
  // std::cout << long_vec.size() << std::endl;
  encoder.encode(long_vec, img_ptxt);
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
  timer.stopTimer(encryptTime); // encryption timer stop

  int t1 = timer.startTimer(); // computation timer start
  // Create rotated copies of the image and multiply by weights
  // std::cout << "Applying Kernel" << std::endl;
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
  timer.stopTimer(t1); // computation timer stop

  int t2 = timer.startTimer(); // decrypt timer start
  // Decrypt & Return result
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(t2); // decrypt timer stop
  return result;
}

//TODO: Need 2x2 expert boxblur example

/// Encrypted BoxBlur, using the pseudocode given by the porcupine paper:
///  Ciphertext boxblur(Ciphertext c0, int h, int w)
///      Ciphertext c1 = rotate(c0, -1 * w)
///      Ciphertext c2 = add(c0, c1)
///      Ciphertext c3 = rotate(c2, -1)
///      return add(c2, c3)
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials
/// \return transformed image
std::vector<int64_t> encryptedBatchedBoxBlur_Porcupine(
        MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree)
{
  /* Setup */
  auto t0 = timer.startTimer(); // keygen timer start

  seal::EncryptionParameters params(seal::scheme_type::bfv);

  params.set_poly_modulus_degree(poly_modulus_degree);
  params.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
  params.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));
  seal::SEALContext context(params);

  // Create keys objects
  seal::KeyGenerator keygen(context);
  seal::SecretKey secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  seal::GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  // Create helper objects
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  timer.stopTimer(t0); // keygen timer stop

  auto encTime = timer.startTimer(); // encryption timer start
  seal::Plaintext plain;
  std::vector<uint64_t> long_vec = std::vector<uint64_t>(img.begin(), img.end());
  batch_encoder.encode(long_vec, plain);

  seal::Ciphertext c0;
  encryptor.encrypt(plain, c0);
  timer.stopTimer(encTime); // encryption timer stop

  /* Computation */
  auto t1 = timer.startTimer();

  // Ciphertext c1 = rotate(c0, -1 * w)
  seal::Ciphertext c1;
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  evaluator.rotate_rows(c0, -1 * imgSize, galois_keys, c1);

  // Ciphertext c2 = add(c0, c1)
  seal::Ciphertext c2;
  evaluator.add(c0, c1, c2);

  // Ciphertext c3 = rotate(c2, -1)
  seal::Ciphertext c3;
  evaluator.rotate_rows(c2, -1, galois_keys, c3);

  // return add(c2, c3)
  seal::Ciphertext result;
  evaluator.add(c2, c3, result);
  timer.stopTimer(t1);

  /* Decrypt */
  auto t2 = timer.startTimer();

  seal::Plaintext decrypted;
  decryptor.decrypt(result, decrypted);

  std::vector<int64_t> retVal;
  batch_encoder.decode(decrypted, retVal);
  timer.stopTimer(t2);
  return retVal;
}

#endif
