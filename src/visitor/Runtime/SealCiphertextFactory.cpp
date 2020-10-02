#include "ast_opt/visitor/Runtime/SealCiphertext.h"
#include "ast_opt/visitor/Runtime/SealCiphertextFactory.h"

void SealCiphertextFactory::setupContext() {
  /// Wrapper for parameters
  seal::EncryptionParameters params(seal::scheme_type::BFV);

  // in BFV, this degree is also the number of slots.
  params.set_poly_modulus_degree(NUM_CTXT_SLOTS);

  // Let SEAL select a "suitable" coefficient modulus (not necessarily optimal)
  params.set_coeff_modulus(seal::CoeffModulus::BFVDefault(params.poly_modulus_degree()));

  // Let SEAL select a plaintext modulus and 20 bit primes that actually support batching
  params.set_plain_modulus(seal::PlainModulus::Batching(params.poly_modulus_degree(), 20));

  // Instantiate context
  context = seal::SEALContext::Create(params);

  /// Helper object to create keys
  seal::KeyGenerator keyGenerator(context);
  secretKey = std::make_unique<seal::SecretKey>(keyGenerator.secret_key());
  publicKey = std::make_unique<seal::PublicKey>(keyGenerator.public_key());
  galoisKeys = std::make_unique<seal::GaloisKeys>(keyGenerator.galois_keys_local());
  relinKeys = std::make_unique<seal::RelinKeys>(keyGenerator.relin_keys_local());

  encoder = std::make_unique<seal::BatchEncoder>(context);
  encryptor = std::make_unique<seal::Encryptor>(context, *publicKey);
  evaluator = std::make_unique<seal::Evaluator>(context);
}

template<typename T>
void SealCiphertextFactory::expandVector(std::vector<T> &values) {
  if (values.size() > encoder->slot_count()) {
    throw std::runtime_error("Cannot encode " + std::to_string(values.size())
                                 + " elements in a ciphertext of size "
                                 + std::to_string(encoder->slot_count()) + ". ");
  }
  // fill vector up to size of ciphertext with last element in given values
  auto lastValue = values.back();
  values.insert(values.end(), encoder->slot_count() - values.size(), lastValue);
}

std::unique_ptr<AbstractCiphertext> SealCiphertextFactory::createCiphertext(std::vector<int64_t> &data) {
  if (!context || !context->parameters_set()) setupContext();
  expandVector(data);

  seal::Plaintext ptxt;
  encoder->encode(data, ptxt);

  std::unique_ptr<SealCiphertext> ctxt = std::make_unique<SealCiphertext>();
  encryptor->encrypt(ptxt, ctxt->ciphertext);

  return ctxt;
}

