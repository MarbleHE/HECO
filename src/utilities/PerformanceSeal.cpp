#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include <chrono>


void bfv_performance_test(seal::SEALContext context)
{
  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;

  auto &parms = context.first_context_data()->parms();
  auto &plain_modulus = parms.plain_modulus();

  size_t poly_modulus_degree = parms.poly_modulus_degree();


  std::cout << "Generating secret/public keys: ";
  time_start = std::chrono::high_resolution_clock::now();
  seal::KeyGenerator keygen(context);
  auto secret_key = keygen.secret_key();
  seal::PublicKey public_key;
  keygen.create_public_key(public_key);
  time_end = std::chrono::high_resolution_clock::now();
  time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);





}

#endif