#ifdef HAVE_SEAL_BFV

#include "ast_opt/utilities/PerformanceSeal.h"

void bfv_performance_test(seal::SEALContext context) {

  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;

  print_parameters(context);
  std::cout << std::endl;

  auto &parms = context.first_context_data()->parms();
  auto &plain_modulus = parms.plain_modulus();

  size_t poly_modulus_degree = parms.poly_modulus_degree();

  // public and secret keys
  std::cout << "Generating secret/public keys: ";
  time_start = std::chrono::high_resolution_clock::now();
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  time_end = std::chrono::high_resolution_clock::now();
  time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  std::cout << "Done [" << time_diff.count() << " microseconds]" << std::endl;

  // relin and galois keys
  seal::RelinKeys relin_keys;
  seal::GaloisKeys gal_keys;
  if (context.using_keyswitching()) {
    /*
    Generate relinearization keys.
    */
    std::cout << "Generating relinearization keys: ";
    time_start = std::chrono::high_resolution_clock::now();
    keygen.create_relin_keys(relin_keys);
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    std::cout << "Done [" << time_diff.count() << " microseconds]" << std::endl;

    if (!context.key_context_data()->qualifiers().using_batching) {
      std::cout << "Given encryption parameters do not support batching." << std::endl;
      return;
    }

    std::cout << "Generating Galois keys: ";
    time_start = std::chrono::high_resolution_clock::now();
    keygen.create_galois_keys(gal_keys);
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    std::cout << "Done [" << time_diff.count() << " microseconds]" << std::endl;
  }

  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);

  /*
   These will hold the total times used by each operation.
   */
  std::chrono::microseconds time_batch_sum(0);
  std::chrono::microseconds time_encrypt_sum(0);
  std::chrono::microseconds time_decrypt_sum(0);
  std::chrono::microseconds time_add_sum(0);
  std::chrono::microseconds time_multiply_1_sum(0);
  std::chrono::microseconds time_multiply_2_sum(0);
  std::chrono::microseconds time_multiply_3_sum(0);
  std::chrono::microseconds time_add_plain_sum(0);
  std::chrono::microseconds time_multiply_plain_sum(0);
  std::chrono::microseconds time_relinearize_sum(0);
  std::chrono::microseconds time_relin_2_sum(0);

  // How many times to run the test?
  long long iterations = 10;
  long long count = 100;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_batch_vec;
  std::vector<std::chrono::microseconds> time_enc_vec;
  std::vector<std::chrono::microseconds> time_dec_vec;
  std::vector<std::chrono::microseconds> time_add_vec;
  std::vector<std::chrono::microseconds> time_mult_vec;
  std::vector<std::chrono::microseconds> time_add_plain_vec;
  std::vector<std::chrono::microseconds> time_multiply_plain_vec;
  std::vector<std::chrono::microseconds> time_relinearize_vec;


  //Populate a vector of values to batch.
  size_t slot_count = batch_encoder.slot_count();
  std::vector<uint64_t> pod_vector;
  std::random_device rd;
  for (size_t i = 0; i < slot_count; i++) {
    pod_vector.push_back(plain_modulus.reduce(rd()));
  }

  std::cout << "Running tests ";

  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      seal::Plaintext plain(poly_modulus_degree, 0);
      seal::Plaintext plain1(poly_modulus_degree, 0);
      seal::Plaintext plain2(poly_modulus_degree, 0);
      seal::Plaintext plain3(poly_modulus_degree, 0);
      seal::Plaintext plain4(poly_modulus_degree, 0);
      seal::Plaintext plain5(poly_modulus_degree, 0);
      seal::Plaintext plain6(poly_modulus_degree, 0);
      seal::Plaintext plain7(poly_modulus_degree, 0);
      seal::Plaintext plain8(poly_modulus_degree, 0);

      // Batching
      time_start = std::chrono::high_resolution_clock::now();
      batch_encoder.encode(pod_vector, plain);
      time_end = std::chrono::high_resolution_clock::now();
      time_batch_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_batch_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));

      // Encryption
      seal::Ciphertext encrypted(context);
      time_start = std::chrono::high_resolution_clock::now();
      encryptor.encrypt(plain, encrypted);
      time_end = std::chrono::high_resolution_clock::now();
      time_encrypt_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_enc_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));

      // Decryption
      time_start = std::chrono::high_resolution_clock::now();
      decryptor.decrypt(encrypted, plain2);
      time_end = std::chrono::high_resolution_clock::now();
      time_decrypt_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_dec_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
      if (plain2!=plain) {
        throw std::runtime_error("Encrypt/decrypt failed. Something is wrong.");
      }

      seal::Ciphertext encrypted1(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain1);
      encryptor.encrypt(plain1, encrypted1);
      seal::Ciphertext encrypted2(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i + 1), plain2);
      encryptor.encrypt(plain2, encrypted2);
      seal::Ciphertext encrypted3(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain3);
      encryptor.encrypt(plain3, encrypted3);
      seal::Ciphertext encrypted4(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain4);
      encryptor.encrypt(plain4, encrypted4);
      seal::Ciphertext encrypted5(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain5);
      encryptor.encrypt(plain5, encrypted5);
      seal::Ciphertext encrypted6(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain6);
      encryptor.encrypt(plain6, encrypted6);
      seal::Ciphertext encrypted7(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain7);
      encryptor.encrypt(plain7, encrypted7);
      seal::Ciphertext encrypted8(context);
      batch_encoder.encode(std::vector<uint64_t>(slot_count, i), plain8);
      encryptor.encrypt(plain8, encrypted8);

      // Ctxt-Ctxt Addition
      time_start = std::chrono::high_resolution_clock::now();
      evaluator.add_inplace(encrypted1, encrypted1);
      evaluator.add_inplace(encrypted2, encrypted2);
      evaluator.add_inplace(encrypted1, encrypted2);
      time_end = std::chrono::high_resolution_clock::now();
      time_add_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_add_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));

      // Ctxt-Ctxt Multiplication
      encrypted1.reserve(3);
      time_start = std::chrono::high_resolution_clock::now();
      evaluator.multiply_inplace(encrypted1, encrypted2);
      time_end = std::chrono::high_resolution_clock::now();
      time_multiply_1_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_mult_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));

      //Relin
      time_start = std::chrono::high_resolution_clock::now();
      evaluator.relinearize_inplace(encrypted1, relin_keys);
      time_end = std::chrono::high_resolution_clock::now();
      time_relinearize_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_relinearize_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));

      // Ptxt-Ctxt Addition
      time_start = std::chrono::high_resolution_clock::now();
      evaluator.add_plain_inplace(encrypted2, plain);
      time_end = std::chrono::high_resolution_clock::now();
      time_add_plain_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_add_plain_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));

      // Ptxt-Ctxt Multiplication
      time_start = std::chrono::high_resolution_clock::now();
      evaluator.multiply_plain_inplace(encrypted2, plain);
      time_end = std::chrono::high_resolution_clock::now();
      time_multiply_plain_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_multiply_plain_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    }
  }

  std::cout << " Done" << std::endl << std::endl;
  std::cout.flush();

  auto avg_batch = time_batch_sum.count()/(count * iterations);
  auto avg_encrypt = time_encrypt_sum.count()/(count * iterations);
  auto avg_decrypt = time_decrypt_sum.count()/(count * iterations);
  auto avg_add = time_add_sum.count()/(3*(count * iterations));
  auto avg_multiply_1 = time_multiply_1_sum.count()/(count * iterations);
  auto avg_add_plain = time_add_plain_sum.count()/(count * iterations);
  auto avg_multiply_plain = time_multiply_plain_sum.count()/(count * iterations);
  auto avg_relin = time_relinearize_sum.count()/(count * iterations);

  std::cout << "Average batch: " << avg_batch << " microseconds" << std::endl;
  std::cout << "Average encrypt: " << avg_encrypt << " microseconds" << std::endl;
  std::cout << "Average decrypt: " << avg_decrypt << " microseconds" << std::endl;
  std::cout << "Average relinearize: " << avg_relin << " microseconds" << std::endl;
  std::cout << "Average add: " << avg_add << " microseconds" << std::endl;
  std::cout << "Average multiply: " << avg_multiply_1 << " microseconds" << std::endl;
  std::cout << "Average add plain: " << avg_add_plain << " microseconds" << std::endl;
  std::cout << "Average multiply plain: " << avg_multiply_plain << " microseconds" << std::endl;

  // Write to csv (result vectors)
  auto &context_data = *context.key_context_data();
  std::string bitsizes_string;
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_modulus_size = coeff_modulus.size();
  for (std::size_t k = 0; k < coeff_modulus_size - 1; k++)
  {
    bitsizes_string = bitsizes_string + std::to_string(coeff_modulus[k].bit_count()) + "+";
  }
  bitsizes_string = bitsizes_string + std::to_string(coeff_modulus.back().bit_count());

  std::string filename = "benchmark_seal_" + std::to_string(poly_modulus_degree) + "_" + bitsizes_string + ".csv";
  std::ofstream myFile(filename); // this will be in cmake-build-debug/test
  std::cout << std::endl;
  std::cout << "Writing to file " << filename << ":";
  myFile << "polymodulus" << " ," << "plainmodulus" << " ," << "coeffmodulus bitsize" << " ," << "bitsizes (q_i)" << " ,"
         << "batch" << " ," << "enc" << " ," << "dec" << " ,"
         << "relin" << " ," << "add" << " ," << "mult" << " ,"
         << "add_plain" << " ," << "mult_plain" << "\n";
  myFile << poly_modulus_degree             << " ," << plain_modulus.value()    << " ," << context_data.total_coeff_modulus_bit_count() << " ,"
         <<  bitsizes_string << " ,"
         << time_batch_vec[0].count()       << " ," << time_enc_vec[0].count()  << " ," << time_dec_vec[0].count()               << " ,"
         << time_relinearize_vec[0].count() << " ," << time_add_vec[0].count()  << " ," << time_mult_vec[0].count()              << " ,"
         << time_add_plain_vec[0].count()   << " ," << time_multiply_plain_vec[0].count() << "\n";
  for (int jj = 1; jj < time_batch_vec.size(); jj++) {
    myFile <<  " ," <<  " ,"  << " ," << " ,"
           << time_batch_vec[jj].count()       << " ," << time_enc_vec[jj].count()  << " ," << time_dec_vec[jj].count()  << " ,"
           << time_relinearize_vec[jj].count() << " ," << time_add_vec[jj].count()  << " ," << time_mult_vec[jj].count() << " ,"
           << time_add_plain_vec[jj].count()   << " ," << time_multiply_plain_vec[jj].count() << "\n";
  }
  myFile.close();
  std::cout << " Done" << std::endl;
  std::cout << std::endl;
}

#endif