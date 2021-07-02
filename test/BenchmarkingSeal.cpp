#include "ast_opt/utilities/PerformanceSeal.h"
#include <memory>

#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV

class BenchmarkingSeal : public ::testing::Test {

 protected:
  const int poly_modulus_degree = 4096;
  seal::SEALContext context;

  void SetUp() override {
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(1024);
    seal::SEALContext context(parms);
  }
};

TEST_F(BenchmarkingSeal, benchmark) {
  bfv_performance_test(context);
}


#endif