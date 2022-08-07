#include "LaplaceSharpening.h"
#include <cmath>
#include "MultiTimer.h"

#ifdef  HAVE_SEAL_BFV
#include "seal/seal.h"
#endif

std::vector<int> laplacianSharpening(const std::vector<int> &img) {
  const auto imgSize = (int) std::ceil(std::sqrt(img.size()));
  std::vector<std::vector<int>> weightMatrix = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  std::vector<int> img2(img.begin(), img.end());
  for (int x = 0; x < imgSize; ++x) {
    for (int y = 0; y < imgSize; ++y) {
      int value = 0;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {
          value += weightMatrix.at(i + 1).at(j + 1)
              *img.at(((x + i)*imgSize + (y + j))%img.size());
        }
      }
      img2[imgSize*x + y] = 2*img[imgSize*x + y] - value;
    }
  }
  return img2;
}

#ifdef HAVE_SEAL_BFV
std::vector<int64_t> encryptedLaplacianSharpening(
    MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights) {
  int t0 = timer.startTimer(); // keygen timer start
  // Input Check
  if (img.size()!=poly_modulus_degree/2) {
    std::cerr << "WARNING: BatchedBoxBlur might be incorrect when img.size() does not match N/2." << std::endl;
  }

  /// Rotations for 3x3 Kernel
  /// Offsets correspond to the different kernel positions
  int img_size = (int) std::sqrt(img.size());
  std::vector<int>
      rotations = {-img_size + 1, 1, img_size + 1, -img_size, 0, img_size, -img_size - 1, -1, img_size - 1};

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
  seal::Evaluator evaluator(context);

  // Create Weight Matrix
  std::vector<int> weight_matrix = {1, 1, 1, 1, -8, 1, 1, 1, 1};
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
  for (size_t i = 0; i < weight_matrix.size(); ++i) {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix[i]), w_ptxts[i]);
    if (encrypt_weights) {
      encryptor.encrypt(w_ptxts[i], w_ctxts[i]);
    }
  }
  timer.stopTimer(encryptTime); // encryption timer stop

  int t1 = timer.startTimer(); // computation timer start
  // Create rotated copies of the image and multiply by weights
  // std::cout << "Applying Kernel" << std::endl;
  std::vector<seal::Ciphertext> rotated_img_ctxts(9, seal::Ciphertext(context));
  for (size_t i = 0; i < rotations.size(); ++i) {
    evaluator.rotate_rows(img_ctxt, rotations[i], galoisKeys, rotated_img_ctxts[i]);

    if (encrypt_weights) {
      evaluator.multiply_inplace(rotated_img_ctxts[i], w_ctxts[i]);
      // relinearization not needed since no more mults coming up
    } else {
      // If the weight is ptxt and one, we can skip this entirely
      if (weight_matrix[i]!=1) {
        evaluator.multiply_plain_inplace(rotated_img_ctxts[i], w_ptxts[i]);
      }
    }
  }

  // Sum up all the ciphertexts
  seal::Ciphertext sum_ctxt(context);
  evaluator.add_many(rotated_img_ctxts, sum_ctxt);

  // Compute img2 = 2*img - value
  seal::Ciphertext result_ctxt;
  evaluator.add(img_ctxt, img_ctxt, result_ctxt);
  evaluator.sub_inplace(result_ctxt, sum_ctxt);

  timer.stopTimer(t1); // computation timer stop

  // Decrypt & Return result
  int t2 = timer.startTimer(); // decrypt timer start
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(t2); // decrypt timer stop
  return result;
}

std::vector<int> encryptedNaiveLaplaceSharpening(
    MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights) {
  auto keygenTimer = timer.startTimer();
  int img_size = (int) std::sqrt(img.size());
  std::vector<int> weight_matrix = {1, 1, 1, 1, -8, 1, 1, 1, 1};

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
  std::vector<seal::Ciphertext> img_ctxt(img.size());
  for (int i = 0; i < img.size(); ++i) {
    uint64_t pixel = img[i];
    seal::Plaintext tmp = seal::Plaintext(seal::util::uint_to_hex_string(&pixel, std::size_t(1)));
    encryptor.encrypt(tmp, img_ctxt[i]);
  }
  // Encode & (if needed) Encrypt the weights
  std::vector<seal::Plaintext> w_ptxts(weight_matrix.size());
  std::vector<seal::Ciphertext> w_ctxts(weight_matrix.size());
  for (size_t i = 0; i < weight_matrix.size(); ++i) {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix[i]), w_ptxts[i]);
    if (encrypt_weights) {
      encryptor.encrypt(w_ptxts[i], w_ctxts[i]);
    }
  }
  timer.stopTimer(encTimer); // encryption timer stop

  //Compute
  auto compTimer = timer.startTimer();
  std::vector<seal::Ciphertext> result_ctxt(img_ctxt.size());

  seal::Plaintext zero_ptxt;
  std::vector<uint64_t> zeros(img_ctxt.size(), 0);
  encoder.encode(zeros, zero_ptxt);
  seal::Ciphertext zero_ctxt;
  encryptor.encrypt(zero_ptxt, zero_ctxt);

  for (int x = 0; x < img_size; ++x) {
    for (int y = 0; y < img_size; ++y) {
      // if we don't initialize this with a valid value, we cannot add_inplace later!
      // TODO: Port this fix to other similar implementations
      seal::Ciphertext value = zero_ctxt;

      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {

          seal::Ciphertext pixel = img_ctxt[((x + i)*img_size + (y + j))%img.size()];

          if (encrypt_weights) {
            evaluator.multiply_inplace(pixel, w_ctxts[i]);
            // relinearization not needed since no more mults coming up
          } else {
            // If the weight is ptxt and one, we can skip this entirely
            if (weight_matrix[i]!=1) {
              // TODO: port the fix for w_pxts[i] going to w_ptxts[-1]!
              evaluator.multiply_plain_inplace(pixel, w_ptxts[(i + 1)*3 + (j + 1)]);
            }
          }

          evaluator.add_inplace(value, pixel);
        }
      }
      seal::Ciphertext two_times;
      evaluator.add(img_ctxt[img_size*x + y], img_ctxt[img_size*x + y], two_times);
      evaluator.sub_inplace(two_times, value);
      result_ctxt[img_size*x + y] = two_times;
    }
  }
  timer.stopTimer(compTimer);

  // Decrypt results
  auto decTimer = timer.startTimer();
  std::vector<int> result(img.size());
  for (int i = 0; i < result.size(); ++i) {
    seal::Plaintext tmp;
    decryptor.decrypt(result_ctxt[i], tmp);
    result[i] = (int) *tmp.data();
  }
  timer.stopTimer(decTimer);
  return result;
}

seal::Ciphertext getIndex(const seal::Ciphertext &vector_ctxt,
                          int i,
                          seal::Evaluator &evaluator,
                          seal::GaloisKeys &galoisKeys,
                          seal::BatchEncoder &encoder) {
  seal::Ciphertext rotated;
  evaluator.rotate_rows(vector_ctxt, -i, galoisKeys, rotated);
  std::vector<int64_t> mask(vector_ctxt.size(), 0);
  mask[0] = 1;
  seal::Plaintext mask_ptxt;
  encoder.encode(mask, mask_ptxt);
  evaluator.multiply_plain_inplace(rotated, mask_ptxt);
  return rotated;
}

void insertInto(seal::Ciphertext &vector_ctxt,
                int i,
                seal::Ciphertext &scalar_ctxt,
                seal::Evaluator &evaluator,
                seal::GaloisKeys &galoisKeys,
                seal::BatchEncoder &encoder) {
  seal::Ciphertext rotated;
  evaluator.rotate_rows(scalar_ctxt, i, galoisKeys, rotated);
  std::vector<int64_t> mask(vector_ctxt.size(), 1);
  mask[i] = 0;
  seal::Plaintext mask_ptxt;
  encoder.encode(mask, mask_ptxt);
  evaluator.multiply_plain_inplace(vector_ctxt, mask_ptxt);
  evaluator.add_inplace(vector_ctxt, rotated);
}

std::vector<int64_t> encryptedNaiveBatchedLaplacianSharpening(
    MultiTimer &timer, const std::vector<int> &img, size_t poly_modulus_degree, bool encrypt_weights) {
  int t0 = timer.startTimer(); // keygen timer start
  // Input Check
  if (img.size()!=poly_modulus_degree/2) {
    std::cerr << "WARNING: BatchedBoxBlur might be incorrect when img.size() does not match N/2." << std::endl;
  }

  /// Rotations for 3x3 Kernel
  /// Offsets correspond to the different kernel positions
  int img_size = (int) std::sqrt(img.size());

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
  keygen.create_galois_keys(galoisKeys);

  // Create helper objects
  seal::BatchEncoder encoder(context);
  seal::Encryptor encryptor(context, publicKey, secretKey);
  seal::Decryptor decryptor(context, secretKey);
  seal::Evaluator evaluator(context);

  //auto at = [&](auto& ctxt, auto index) { return getIndex(ctxt, index, evaluator, galoisKeys, encoder); };

  // Create Weight Matrix
  std::vector<int> weight_matrix = {1, 1, 1, 1, -8, 1, 1, 1, 1};
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
  for (size_t i = 0; i < weight_matrix.size(); ++i) {
    encoder.encode(std::vector<int64_t>(encoder.slot_count(), weight_matrix[i]), w_ptxts[i]);
    if (encrypt_weights) {
      encryptor.encrypt(w_ptxts[i], w_ctxts[i]);
    }
  }
  timer.stopTimer(encryptTime); // encryption timer stop

  //Compute
  auto compTimer = timer.startTimer();
  seal::Ciphertext result_ctxt = img_ctxt;

  seal::Plaintext zero_ptxt;
  std::vector<uint64_t> zeros(img_ctxt.size(), 0);
  encoder.encode(zeros, zero_ptxt);
  seal::Ciphertext zero_ctxt;
  encryptor.encrypt(zero_ptxt, zero_ctxt);

  for (int x = 0; x < img_size; ++x) {
    for (int y = 0; y < img_size; ++y) {
      seal::Ciphertext value = zero_ctxt;
      for (int j = -1; j < 2; ++j) {
        for (int i = -1; i < 2; ++i) {

          seal::Ciphertext
              pixel = getIndex(img_ctxt, ((x + i)*img_size + (y + j))%img.size(), evaluator, galoisKeys, encoder);

          if (encrypt_weights) {
            evaluator.multiply_inplace(pixel, w_ctxts[i]);
            // relinearization not needed since no more mults coming up
          } else {
            // If the weight is ptxt and one, we can skip this entirely
            if (weight_matrix[i]!=1) {
              evaluator.multiply_plain_inplace(pixel, w_ptxts[(i + 1)*3 + (j + 1)]);
            }
          }

          evaluator.add_inplace(value, pixel);
        }
      }
      seal::Ciphertext two_times;
      auto orig_pixel = getIndex(img_ctxt, img_size*x + y, evaluator, galoisKeys, encoder);
      evaluator.add(orig_pixel, orig_pixel, two_times);
      evaluator.sub_inplace(two_times, value);
      insertInto(result_ctxt, img_size*x + y, two_times, evaluator, galoisKeys, encoder);
    }
  }
  timer.stopTimer(compTimer);

  // Decrypt & Return result
  int t2 = timer.startTimer(); // decrypt timer start
  seal::Plaintext result_ptxt;
  decryptor.decrypt(result_ctxt, result_ptxt);
  std::vector<int64_t> result;
  encoder.decode(result_ptxt, result);
  timer.stopTimer(t2); // decrypt timer stop
  return result;
}
#endif