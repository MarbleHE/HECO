#include "ast_opt/utilities/PerformanceSeal.h"
#include <memory>
#include <algorithm>

#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV

class BenchmarkingSeal : public ::testing::Test {

 protected:
  const int poly_modulus_degree = 8192;

  void SetUp() override {

  }
};

TEST_F(BenchmarkingSeal, benchmark) {

 // for (int i = 0; i < 1; i++) {
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    //parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    //parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(
      //  poly_modulus_degree, seal::sec_level_type::tc128));
    std::vector<int> bitsizes =  {60, 60, 60};
    parms.set_coeff_modulus(seal::CoeffModulus::Create(
       poly_modulus_degree,  bitsizes));
    parms.set_plain_modulus(seal::PlainModulus::Batching(parms.poly_modulus_degree(), 20));
    seal::SEALContext context(parms);
    bfv_performance_test(context);
 // }
}


TEST_F(BenchmarkingSeal, noModSwitchTest) {

  // set up seal context
  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(parms.poly_modulus_degree()));
  //parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(
    //  poly_modulus_degree, seal::sec_level_type::tc128));
  parms.set_plain_modulus(seal::PlainModulus::Batching(parms.poly_modulus_degree(), 20));
  seal::SEALContext context(parms);

  // print params
  print_parameters(context);
  std::cout << std::endl;

  // public and secret keys
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);

  // relin and galois keys
  seal::RelinKeys relin_keys;
  seal::GaloisKeys gal_keys;
  keygen.create_relin_keys(relin_keys);
  keygen.create_galois_keys(gal_keys);

  // encryptor etc
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);

  // time var
  std::chrono::microseconds time_sum(0);

  // How many times to run the test?
  long long iterations = 1;
  long long count = 1000;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("2x^3 + 3x^2 + 3x^1 + 4");
  seal::Plaintext zPlain("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);
  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);
  seal::Ciphertext zEncrypted;
  encryptor.encrypt(zPlain, zEncrypted);

  //ctxt variables
  seal::Ciphertext xPow2;
  seal::Ciphertext xPow3;
  seal::Ciphertext xPow4;
  seal::Ciphertext zPow2;
  seal::Ciphertext zPow3;
  seal::Ciphertext zPow4;
  seal::Ciphertext xPow4Plusy;
  seal::Ciphertext result;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;


  //compute (x^4 + y) * z^4 WITHOUT modswitch before last mult

  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      //start timer
      time_start = std::chrono::high_resolution_clock::now();

      // x^4
      evaluator.multiply(xEncrypted, xEncrypted, xPow2);
      evaluator.multiply(xPow2, xEncrypted, xPow3);
      evaluator.multiply(xPow3, xEncrypted, xPow4);

      //z^4
      evaluator.multiply(zEncrypted, zEncrypted, zPow2);
      evaluator.multiply(zPow2, zEncrypted, zPow3);
      evaluator.multiply(zPow3, zEncrypted, zPow4);

      //x^4 + y
      evaluator.add(xPow4, yEncrypted, xPow4Plusy);

      // (x^4 + y) * z^4
      evaluator.multiply(xPow4Plusy, zPow4, result);

      time_end = std::chrono::high_resolution_clock::now();
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
      time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);

      //std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count()
        //<< std::endl;

    }
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count * iterations);

  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of (x^4 + y) * z^4 WITHOUT modswitch [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;
}

TEST_F(BenchmarkingSeal, modSwitchTest) {

  // set up seal context
  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(
      poly_modulus_degree, seal::sec_level_type::tc128));
  parms.set_plain_modulus(seal::PlainModulus::Batching(parms.poly_modulus_degree(), 20));
  seal::SEALContext context(parms);

  // print params
  print_parameters(context);
  std::cout << std::endl;

  // public and secret keys
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);

  // relin and galois keys
  seal::RelinKeys relin_keys;
  seal::GaloisKeys gal_keys;
  keygen.create_relin_keys(relin_keys);
  keygen.create_galois_keys(gal_keys);

  // encryptor etc
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);

  // time var
  std::chrono::microseconds time_sum(0);

  // How many times to run the test?
  long long iterations = 1;
  long long count = 1000;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("2x^3 + 3x^2 + 3x^1 + 4");
  seal::Plaintext zPlain("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);
  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);
  seal::Ciphertext zEncrypted;
  encryptor.encrypt(zPlain, zEncrypted);

  //ctxt variables
  seal::Ciphertext xPow2;
  seal::Ciphertext xPow3;
  seal::Ciphertext xPow4;
  seal::Ciphertext zPow2;
  seal::Ciphertext zPow3;
  seal::Ciphertext zPow4;
  seal::Ciphertext xPow4Plusy;
  seal::Ciphertext result;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;


  //compute (x^4 + y) * z^4 WITH modswitch before last mult

  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      // start timer
      time_start = std::chrono::high_resolution_clock::now();

      // x^4
      evaluator.multiply(xEncrypted, xEncrypted, xPow2);
      evaluator.multiply(xPow2, xEncrypted, xPow3);
      evaluator.multiply(xPow3, xEncrypted, xPow4);

      //z^4
      evaluator.multiply(zEncrypted, zEncrypted, zPow2);
      evaluator.multiply(zPow2, zEncrypted, zPow3);
      evaluator.multiply(zPow3, zEncrypted, zPow4);

      //x^4 + y
      evaluator.add(xPow4, yEncrypted, xPow4Plusy);

      //mod switch z^4 and (x^4 + y)
      evaluator.mod_switch_to_next_inplace(zPow4);
      evaluator.mod_switch_to_next_inplace(xPow4Plusy);

      // (x^4 + y) * z^4
      evaluator.multiply(xPow4Plusy, zPow4, result);

      time_end = std::chrono::high_resolution_clock::now();
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_diff));
      time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_diff);

    //  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count()
      //  << std::endl;

    }
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count * iterations);

  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of (x^4 + y) * z^4 WITH modswitch [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;
}


TEST_F(BenchmarkingSeal, xPow4noConeRewr) {

  // set up seal context
  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(
      poly_modulus_degree, seal::sec_level_type::tc128));
  parms.set_plain_modulus(seal::PlainModulus::Batching(parms.poly_modulus_degree(), 20));
  seal::SEALContext context(parms);

  // print params
  print_parameters(context);
  std::cout << std::endl;

  // public and secret keys
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);

  // relin and galois keys
  seal::RelinKeys relin_keys;
  seal::GaloisKeys gal_keys;
  keygen.create_relin_keys(relin_keys);
  keygen.create_galois_keys(gal_keys);

  // encryptor etc
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);

  // time var
  std::chrono::microseconds time_sum(0);

  // How many times to run the test?
  long long iterations = 1;
  long long count = 1000;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);


  //ctxt variables
  seal::Ciphertext xPow2;
  seal::Ciphertext xPow3;
  seal::Ciphertext xPow4bad;
  seal::Ciphertext xPow4good;
  seal::Ciphertext result;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;


  //compute (x^4 + y) * z^4 WITH modswitch before last mult

  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      // start timer
      time_start = std::chrono::high_resolution_clock::now();

      // x^4 = x*x*x*x
      evaluator.multiply(xEncrypted, xEncrypted, xPow2);
      evaluator.multiply(xPow2, xEncrypted, xPow3);
      evaluator.multiply(xPow3, xEncrypted, xPow4bad);

      //x^4 = x^2 * x^2
      evaluator.multiply(xPow2, xPow2, xPow4good);

      // (x*x*x*x) * (x^2 * x^2)
      evaluator.multiply(xPow4bad, xPow4good, result);

      time_end = std::chrono::high_resolution_clock::now();
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_diff));
      time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_diff);

      //  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count()
      //  << std::endl;

    }
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count * iterations);

  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of (x*x*x*x) * (x^2 *x^2) [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;
}


TEST_F(BenchmarkingSeal, xPow4afterConeRewr) {

  // set up seal context
  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(
      poly_modulus_degree, seal::sec_level_type::tc128));
  parms.set_plain_modulus(seal::PlainModulus::Batching(parms.poly_modulus_degree(), 20));
  seal::SEALContext context(parms);

  // print params
  print_parameters(context);
  std::cout << std::endl;

  // public and secret keys
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);

  // relin and galois keys
  seal::RelinKeys relin_keys;
  seal::GaloisKeys gal_keys;
  keygen.create_relin_keys(relin_keys);
  keygen.create_galois_keys(gal_keys);

  // encryptor etc
  seal::Encryptor encryptor(context, public_key);
  seal::Decryptor decryptor(context, secret_key);
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batch_encoder(context);

  // time var
  std::chrono::microseconds time_sum(0);

  // How many times to run the test?
  long long iterations = 1;
  long long count = 1000;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);


  //ctxt variables
  seal::Ciphertext xPow2;
  seal::Ciphertext xPow4good;
  seal::Ciphertext result;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;


  //compute (x^4 + y) * z^4 WITH modswitch before last mult

  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      // start timer
      time_start = std::chrono::high_resolution_clock::now();

      // x^4 = x*x*x*x
      evaluator.multiply(xEncrypted, xEncrypted, xPow2);


      //x^4 = x^2 * x^2
      evaluator.multiply(xPow2, xPow2, xPow4good);

      // (x*x*x*x) * (x^2 * x^2)
      evaluator.multiply(xPow4good, xPow4good, result);

      time_end = std::chrono::high_resolution_clock::now();
      time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
      time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_diff));
      time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_diff);

      //  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count()
      //  << std::endl;

    }
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count * iterations);

  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of (x^2 * x^2) * (x^2 * x^2) [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;
}

#endif