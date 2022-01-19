//TODO: Fix and formalize this "header" for the emitC stuff
#include <SEAL-3.6/seal/ciphertext.h>
#include <SEAL-3.6/seal/evaluator.h>
#include <SEAL-3.6/seal/context.h>
#include <SEAL-3.6/seal/galoiskeys.h>

class CustomEvaluator {
 public:
  seal::Evaluator seal_evaluator;
  seal::SEALContext seal_context;
  seal::GaloisKeys seal_galoiskeys;

  inline seal::Ciphertext sub(seal::Ciphertext &a, seal::Ciphertext& b) {
    seal::Ciphertext dst(seal_context);
    seal_evaluator.sub(a,b,dst);
    return dst;
  }
  //inline seal::Ciphertext sub_many(seal::Ciphertext &a, seal::Ciphertext& b) {
  //  seal::Ciphertext dst(seal_context);
  //  seal_evaluator.sub_many(a,b,dst);
  //  return dst;
  //}
  inline seal::Ciphertext add(seal::Ciphertext &a, seal::Ciphertext& b) {
    seal::Ciphertext dst(seal_context);
    seal_evaluator.add(a,b,dst);
    return dst;
  }
  //inline seal::Ciphertext add_many(seal::Ciphertext &a, seal::Ciphertext& b) {
  //  seal::Ciphertext dst(seal_context);
  //  seal_evaluator.add(a,b,dst);
  //  return dst;
  //}
  inline seal::Ciphertext multiply(seal::Ciphertext &a, seal::Ciphertext& b) {
    seal::Ciphertext dst(seal_context);
    seal_evaluator.multiply(a,b,dst);
    return dst;
  }
  //inline seal::Ciphertext multiply_many(seal::Ciphertext &a, seal::Ciphertext& b) {
  //  seal::Ciphertext dst(seal_context);
  //  seal_evaluator.multiply(a,b,dst);
  //  return dst;
  //}
  inline seal::Ciphertext rotate(seal::Ciphertext &x, int i) {
    seal::Ciphertext dst(seal_context);
    seal_evaluator.rotate_vector(x,i,seal_galoiskeys,i);
  }

};
//TODO: Initialize stuff
CustomEvaluator evaluator;

seal::Ciphertext encryptedHammingDistance(seal::Ciphertext v1, seal::Ciphertext v2) {
  seal::Ciphertext v3 = evaluator.sub(v1, v2);
  seal::Ciphertext v4 = evaluator.multiply(v3, v3);
  seal::Ciphertext v5 = evaluator.rotate(v4, -2);
  seal::Ciphertext v6 = evaluator.add(v4, v5);
  seal::Ciphertext v7 = evaluator.rotate(v6, -1);
  seal::Ciphertext v8 = evaluator.add(v6, v7);
  seal::Ciphertext v9 = evaluator.rotate(v8, 0);
  return v9;
}
