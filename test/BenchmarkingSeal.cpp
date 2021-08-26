#include "ast_opt/utilities/PerformanceSeal.h"
#include <memory>
#include <algorithm>

#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV

class BenchmarkingSeal : public ::testing::Test {

 protected:
  const int poly_modulus_degree = 32768;

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

  /// Seal eval of the circuit evaluating (x^4 + y) * z^4 without modswitch

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
  long long count = 100;

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

  // write to file

  std::cout << poly_modulus_degree << " , " << "(x^4 + y) * z^4 : NO MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }

}

TEST_F(BenchmarkingSeal, modSwitchTest) {

  /// Seal eval of the circuit evaluating (x^4 + y) * z^4 WITH modswitch applied to operands of last mult

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
  long long count = 100;

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

  // write to file

  std::cout << poly_modulus_degree << " , " << "(x^4 + y) * z^4 : MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }

}

TEST_F(BenchmarkingSeal, sum_x_i_Times_yPow8_WITH_MODSWITCH) {

  /// Benchmarking the circuit evaluating \sum (x_i)^2 * y^8: we expect significant speedup when modswitching before taking squares
  /// We modswitch all the way down to the last prime. This is possible because we do a^6 i.e enough noise will be spent

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
  long long count = 100;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext x1Plain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext x2Plain("2x^3 + 3x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Ciphertext x1Encrypted;
  encryptor.encrypt(x1Plain, x1Encrypted);
  seal::Ciphertext x2Encrypted;
  encryptor.encrypt(x2Plain, x2Encrypted);

  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);

  //ctxt variables
  seal::Ciphertext sum;
  seal::Ciphertext x1s, x2s, y2, y3, y4, y5, y6, y7, y8;
  seal::Ciphertext x1Pow2, x2Pow2;
  seal::Ciphertext result;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;


  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      // start timer
      time_start = std::chrono::high_resolution_clock::now();


      // modswitch x_i's
      evaluator.mod_switch_to_next(x1Encrypted, x1s);
      evaluator.mod_switch_to_next(x1Encrypted, x2s);


      // modswitch until the last prime in the chain
      auto &context_data = *context.key_context_data();
      auto coeff_modulus = context_data.parms().coeff_modulus();
      std::size_t coeff_modulus_size = coeff_modulus.size();
      for (int k = 1; k < coeff_modulus_size - 2; k++) {
        evaluator.mod_switch_to_next_inplace(x1s);
        evaluator.mod_switch_to_next_inplace(x2s);
      }


      // square modswitched xi's
      evaluator.multiply(x1s, x1s, x1Pow2);
      evaluator.multiply(x2s, x2s, x2Pow2);

      std::vector<seal::Ciphertext> xi;
      xi.push_back(x1Pow2);
      xi.push_back(x2Pow2);

      // add modswitched xi's
      evaluator.add_many(xi, sum);

      // y^8
      evaluator.multiply(yEncrypted, yEncrypted, y2);
      evaluator.multiply(yEncrypted, y2, y3);
      evaluator.multiply(yEncrypted, y3, y4);
      evaluator.multiply(yEncrypted, y4, y5);
      evaluator.multiply(yEncrypted, y5, y6);
      evaluator.multiply(yEncrypted, y6, y7);
      evaluator.multiply(yEncrypted, y7, y8);

      for (int k = 0; k < coeff_modulus_size - 2; k++) {
        evaluator.mod_switch_to_next_inplace(y8);
      }

      evaluator.multiply(y8, sum, result);

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

  std::cout << "Average evaluation time [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


  // write to file
  std::cout << poly_modulus_degree << " , " << "sum_x_i_Times_yPow8: MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }


}

TEST_F(BenchmarkingSeal, sum_x_i_Times_yPow8_WITHOUT_MODSWITCH) {

  /// Benchmarking the circuit evaluating \sum (x_i)^2 * a^8 without modswitch

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
  long long count = 100;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext x1Plain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext x2Plain("2x^3 + 3x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Ciphertext x1Encrypted;
  encryptor.encrypt(x1Plain, x1Encrypted);
  seal::Ciphertext x2Encrypted;
  encryptor.encrypt(x2Plain, x2Encrypted);

  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);

  //ctxt variables
  seal::Ciphertext sum;
  seal::Ciphertext x1s, x2s, y2, y3, y4, y5, y6, y7, y8;
  seal::Ciphertext x1Pow2, x2Pow2;
  seal::Ciphertext result;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;


  for(size_t j = 0; j < static_cast<size_t>(iterations); j++) {

    for (size_t i = 0; i < static_cast<size_t>(count); i++) {

      // start timer
      time_start = std::chrono::high_resolution_clock::now();

      // square all xi's
      evaluator.multiply(x1Encrypted, x1Encrypted, x1Pow2);
      evaluator.multiply(x2Encrypted, x2Encrypted, x2Pow2);

      std::vector<seal::Ciphertext> xi;
      xi.push_back(x1Pow2);
      xi.push_back(x2Pow2);

      // add modswitched xi's
      evaluator.add_many(xi, sum);


      // y^8
      evaluator.multiply(yEncrypted, yEncrypted, y2);
      evaluator.multiply(yEncrypted, y2, y3);
      evaluator.multiply(yEncrypted, y3, y4);
      evaluator.multiply(yEncrypted, y4, y5);
      evaluator.multiply(yEncrypted, y5, y6);
      evaluator.multiply(yEncrypted, y6, y7);
      evaluator.multiply(yEncrypted, y7, y8);


      evaluator.multiply(y8, sum, result);

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

  std::cout << "Average evaluation time [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  // write to file
  std::cout << poly_modulus_degree << " , " << "sum_x_i_Times_yPow8: NO MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }

}

TEST_F(BenchmarkingSeal, many_adds_WITHMODSWITCH) {

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
  long long count = 100;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("2x^3 + 3x^2 + 3x^1 + 4");
  seal::Plaintext zPlain("3x^3 + 2x^2 + 3x^1 + 4");

  seal::Plaintext a1("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a2("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a3("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a4("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a5("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a6("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a7("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a8("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a9("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a10("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a11("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a12("3x^3 + 2x^2 + 3x^1 + 4");


  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);
  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);
  seal::Ciphertext zEncrypted;
  encryptor.encrypt(zPlain, zEncrypted);

  seal::Ciphertext a1Encrypted;
  encryptor.encrypt(a1, a1Encrypted);
  seal::Ciphertext a2Encrypted;
  encryptor.encrypt(a2, a2Encrypted);
  seal::Ciphertext a3Encrypted;
  encryptor.encrypt(a3, a3Encrypted);
  seal::Ciphertext a4Encrypted;
  encryptor.encrypt(a4, a4Encrypted);
  seal::Ciphertext a5Encrypted;
  encryptor.encrypt(a5, a5Encrypted);
  seal::Ciphertext a6Encrypted;
  encryptor.encrypt(a6, a6Encrypted);
  seal::Ciphertext a7Encrypted;
  encryptor.encrypt(a7, a7Encrypted);
  seal::Ciphertext a8Encrypted;
  encryptor.encrypt(a8, a8Encrypted);
  seal::Ciphertext a9Encrypted;
  encryptor.encrypt(a9, a9Encrypted);
  seal::Ciphertext a10Encrypted;
  encryptor.encrypt(a10, a10Encrypted);
  seal::Ciphertext a11Encrypted;
  encryptor.encrypt(a11, a11Encrypted);
  seal::Ciphertext a12Encrypted;
  encryptor.encrypt(a12, a12Encrypted);

  //ctxt variables
  seal::Ciphertext xPow2;
  seal::Ciphertext xPow3;
  seal::Ciphertext xPow4;
  seal::Ciphertext zPow2;
  seal::Ciphertext zPow3;
  seal::Ciphertext zPow4;
  seal::Ciphertext xPow4Plusy;
  seal::Ciphertext a1s;
  seal::Ciphertext a2s;
  seal::Ciphertext a3s;
  seal::Ciphertext a4s;
  seal::Ciphertext a5s;
  seal::Ciphertext a6s;
  seal::Ciphertext a7s;
  seal::Ciphertext a8s;
  seal::Ciphertext a9s;
  seal::Ciphertext a10s;
  seal::Ciphertext a11s;
  seal::Ciphertext a12s;
  seal::Ciphertext result1;


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

      // result1 = (x^4 + y) * z^4
      evaluator.multiply(xPow4Plusy, zPow4, result1);

      // need to modswitch all a_i's to avoid param mismatch: this should cause longer runtimes
      evaluator.mod_switch_to_next(a1Encrypted, a1s);
      evaluator.mod_switch_to_next(a2Encrypted, a2s);
      evaluator.mod_switch_to_next(a3Encrypted, a3s);
      evaluator.mod_switch_to_next(a4Encrypted, a4s);
      evaluator.mod_switch_to_next(a5Encrypted, a5s);
      evaluator.mod_switch_to_next(a6Encrypted, a6s);
      evaluator.mod_switch_to_next(a7Encrypted, a7s);
      evaluator.mod_switch_to_next(a8Encrypted, a8s);
      evaluator.mod_switch_to_next(a9Encrypted, a9s);
      evaluator.mod_switch_to_next(a10Encrypted, a10s);
      evaluator.mod_switch_to_next(a11Encrypted, a11s);
      evaluator.mod_switch_to_next(a12Encrypted, a12s);


      // add them to result one-by-one
      evaluator.add_inplace(result1,a1s);
      evaluator.add_inplace(result1,a2s);
      evaluator.add_inplace(result1,a3s);
      evaluator.add_inplace(result1,a4s);
      evaluator.add_inplace(result1,a5s);
      evaluator.add_inplace(result1,a6s);
      evaluator.add_inplace(result1,a7s);
      evaluator.add_inplace(result1,a8s);
      evaluator.add_inplace(result1,a9s);
      evaluator.add_inplace(result1,a10s);
      evaluator.add_inplace(result1,a11s);
      evaluator.add_inplace(result1,a12s);




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

  std::cout << "Average evaluation time of (x^4 + y) * z^4 + a1 + ... + a12 WITH modswitch [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  // write to file
  std::cout << poly_modulus_degree << " , " << "many additions: MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }
}

TEST_F(BenchmarkingSeal, many_adds_WITHOUTMODSWITCH) {

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
  long long count = 100;

  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  // encrypt vars
  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("2x^3 + 3x^2 + 3x^1 + 4");
  seal::Plaintext zPlain("3x^3 + 2x^2 + 3x^1 + 4");

  seal::Plaintext a1("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a2("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a3("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a4("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a5("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a6("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a7("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a8("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a9("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a10("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a11("3x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext a12("3x^3 + 2x^2 + 3x^1 + 4");


  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);
  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);
  seal::Ciphertext zEncrypted;
  encryptor.encrypt(zPlain, zEncrypted);

  seal::Ciphertext a1Encrypted;
  encryptor.encrypt(a1, a1Encrypted);
  seal::Ciphertext a2Encrypted;
  encryptor.encrypt(a2, a2Encrypted);
  seal::Ciphertext a3Encrypted;
  encryptor.encrypt(a3, a3Encrypted);
  seal::Ciphertext a4Encrypted;
  encryptor.encrypt(a4, a4Encrypted);
  seal::Ciphertext a5Encrypted;
  encryptor.encrypt(a5, a5Encrypted);
  seal::Ciphertext a6Encrypted;
  encryptor.encrypt(a6, a6Encrypted);
  seal::Ciphertext a7Encrypted;
  encryptor.encrypt(a7, a7Encrypted);
  seal::Ciphertext a8Encrypted;
  encryptor.encrypt(a8, a8Encrypted);
  seal::Ciphertext a9Encrypted;
  encryptor.encrypt(a9, a9Encrypted);
  seal::Ciphertext a10Encrypted;
  encryptor.encrypt(a10, a10Encrypted);
  seal::Ciphertext a11Encrypted;
  encryptor.encrypt(a11, a11Encrypted);
  seal::Ciphertext a12Encrypted;
  encryptor.encrypt(a12, a12Encrypted);

  //ctxt variables
  seal::Ciphertext xPow2;
  seal::Ciphertext xPow3;
  seal::Ciphertext xPow4;
  seal::Ciphertext zPow2;
  seal::Ciphertext zPow3;
  seal::Ciphertext zPow4;
  seal::Ciphertext xPow4Plusy;
  seal::Ciphertext a1s;
  seal::Ciphertext a2s;
  seal::Ciphertext a3s;
  seal::Ciphertext a4s;
  seal::Ciphertext a5s;
  seal::Ciphertext a6s;
  seal::Ciphertext a7s;
  seal::Ciphertext a8s;
  seal::Ciphertext a9s;
  seal::Ciphertext a10s;
  seal::Ciphertext a11s;
  seal::Ciphertext a12s;
  seal::Ciphertext result1;


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

      // result1 = (x^4 + y) * z^4
      evaluator.multiply(xPow4Plusy, zPow4, result1);

      // add them to result one-by-one
      evaluator.add_inplace(result1,a1Encrypted);
      evaluator.add_inplace(result1,a2Encrypted);
      evaluator.add_inplace(result1,a3Encrypted);
      evaluator.add_inplace(result1,a4Encrypted);
      evaluator.add_inplace(result1,a5Encrypted);
      evaluator.add_inplace(result1,a6Encrypted);
      evaluator.add_inplace(result1,a7Encrypted);
      evaluator.add_inplace(result1,a8Encrypted);
      evaluator.add_inplace(result1,a9Encrypted);
      evaluator.add_inplace(result1,a10Encrypted);
      evaluator.add_inplace(result1,a11Encrypted);
      evaluator.add_inplace(result1,a12Encrypted);


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

  std::cout << "Average evaluation time of (x^4 + y) * z^4 + a1 + ... + a12 WITHOUT modswitch [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


  // write to file
  std::cout << poly_modulus_degree << " , " << "many additions: no MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }
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

TEST_F(BenchmarkingSeal, MultDifferentLevels) {

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
  long long count = 1000;


  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("2x^3 + 3x^2 + 3x^1 + 4");

  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);
  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);

  //ctxt variables
  seal::Ciphertext xTimesy;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start;
  std::chrono::high_resolution_clock::time_point time_end;
  std::chrono::microseconds time_diff(0);

  // benchmark mult Level 1
  std::cout << "Benchmarking multiplication Level 1" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.multiply(xEncrypted, yEncrypted, xTimesy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count);
  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Mult Level 1 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


  ///-------------------

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level2(0);

  // modswitch to level 2
  evaluator.mod_switch_to_next_inplace(xEncrypted);
  evaluator.mod_switch_to_next_inplace(yEncrypted);

  // benchmark mult Level 2
  std::cout << "Benchmarking multiplication Level 2" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.multiply(xEncrypted, yEncrypted, xTimesy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level2 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level2).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Mult Level 2 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  //-----------------------

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level3(0);

  // modswitch to level 3
  evaluator.mod_switch_to_next_inplace(xEncrypted);
  evaluator.mod_switch_to_next_inplace(yEncrypted);

  // benchmark mult Level 3
  std::cout << "Benchmarking multiplication Level 3" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.multiply(xEncrypted, yEncrypted, xTimesy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level3 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level3).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Mult Level 3 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


  //-----------------------

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level4(0);

  // modswitch to level 4
  evaluator.mod_switch_to_next_inplace(xEncrypted);
  evaluator.mod_switch_to_next_inplace(yEncrypted);

  // benchmark mult Level 4
  std::cout << "Benchmarking multiplication Level 4" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.multiply(xEncrypted, yEncrypted, xTimesy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level4 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level4).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Mult Level 4 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


}

TEST_F(BenchmarkingSeal, AddDifferentLevels) {

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
  long long count = 1000;


  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");
  seal::Plaintext yPlain("2x^3 + 3x^2 + 3x^1 + 4");

  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);
  seal::Ciphertext yEncrypted;
  encryptor.encrypt(yPlain, yEncrypted);

  //ctxt variables
  seal::Ciphertext xPlusy;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start;
  std::chrono::high_resolution_clock::time_point time_end;
  std::chrono::microseconds time_diff(0);

  // benchmark add Level 1
  std::cout << "Benchmarking addition Level 1" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.add(xEncrypted, yEncrypted, xPlusy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count);
  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Add Level 1 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


  ///-------------------

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level2(0);

  // modswitch to level 2
  evaluator.mod_switch_to_next_inplace(xEncrypted);
  evaluator.mod_switch_to_next_inplace(yEncrypted);

  // benchmark add Level 2
  std::cout << "Benchmarking addition Level 2" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.add(xEncrypted, yEncrypted, xPlusy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level2 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level2).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Add Level 2 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  //-----------------------

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level3(0);

  // modswitch to level 3
  evaluator.mod_switch_to_next_inplace(xEncrypted);
  evaluator.mod_switch_to_next_inplace(yEncrypted);

  // benchmark add Level 3
  std::cout << "Benchmarking addition Level 3" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.add(xEncrypted, yEncrypted, xPlusy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level3 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level3).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Add Level 3 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;


  //-----------------------

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level4(0);

  // modswitch to level 4
  evaluator.mod_switch_to_next_inplace(xEncrypted);
  evaluator.mod_switch_to_next_inplace(yEncrypted);

  // benchmark add Level 4
  std::cout << "Benchmarking addition Level 4" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    evaluator.add(xEncrypted, yEncrypted, xPlusy);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level4 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
  }

  avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level4).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of Add Level 4 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;
}

TEST_F(BenchmarkingSeal, ModSwitchToDifferentLevels) {

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
  long long count = 1000;


  // Vectors holding results of each round
  std::vector<std::chrono::microseconds> time_vec;

  seal::Plaintext xPlain("1x^3 + 2x^2 + 3x^1 + 4");

  seal::Ciphertext xEncrypted;
  encryptor.encrypt(xPlain, xEncrypted);


  //ctxt variables
  seal::Ciphertext xModswitched;
  seal::Ciphertext xModswitched2;
  seal::Ciphertext xModswitched3;
  seal::Ciphertext xModswitched4;

  //timing vars
  std::chrono::high_resolution_clock::time_point time_start;
  std::chrono::high_resolution_clock::time_point time_end;
  std::chrono::microseconds time_diff(0);

  //----

  // benchmark Modswitch Level 1
  std::cout << "Benchmarking Modswitch Level 1" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    // do modswitch
    evaluator.mod_switch_to_next(xEncrypted, xModswitched);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    //std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << std::endl;
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count);
  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time of ModSwitching to Level2 [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  // -----

  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level2(0);


  // benchmark Modswitch Level 1
  std::cout << "Benchmarking Modswitch Level 2" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    // do modswitch
    evaluator.mod_switch_to_next(xModswitched, xModswitched2);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level2 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    //std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << std::endl;
  }

  long long avg_time2 = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level2).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time2) * (time_vec[i].count() - avg_time2);
  }

  std::cout << "Average evaluation time of ModSwitching to Level3 [" << avg_time2 << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

 //---


  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level3(0);


  // benchmark Modswitch Level 1
  std::cout << "Benchmarking Modswitch Level 3" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    // do modswitch
    evaluator.mod_switch_to_next(xModswitched2, xModswitched3);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level3 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    //std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << std::endl;
  }

  long long avg_time3 = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level3).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time3) * (time_vec[i].count() - avg_time3);
  }

  std::cout << "Average evaluation time of ModSwitching to Level3 [" << avg_time3 << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  //---


  // clear time_vec and time_sum
  time_vec.clear();
  std::chrono::microseconds time_sum_level4(0);


  // benchmark Modswitch Level 1
  std::cout << "Benchmarking Modswitch Level 4" << std::endl;

  for(size_t j = 0; j < static_cast<size_t>(count); j++) {

    //start timer
    time_start = std::chrono::high_resolution_clock::now();

    // do modswitch
    evaluator.mod_switch_to_next(xModswitched2, xModswitched3);

    // end timer
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start));
    time_sum_level4 += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    //std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << std::endl;
  }

  long long avg_time4 = std::chrono::duration_cast<std::chrono::microseconds>(time_sum_level4).count()/(count);
  //calc std deviation
  standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time4) * (time_vec[i].count() - avg_time4);
  }

  std::cout << "Average evaluation time of ModSwitching to Level4 [" << avg_time4 << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  //---


}

#endif