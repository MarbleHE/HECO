#include "ast_opt/utilities/PerformanceSeal.h"
#include <memory>
#include <algorithm>

#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV

class BenchmarkingSeal : public ::testing::Test {

 protected:
  const int poly_modulus_degree = 16384;

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
    std::vector<int> bitsizes =  {60, 60, 30, 30, 30, 30, 30, 30, 30};
    parms.set_coeff_modulus(seal::CoeffModulus::Create(
       poly_modulus_degree,  bitsizes));
    parms.set_plain_modulus(seal::PlainModulus::Batching(parms.poly_modulus_degree(), 20));
    seal::SEALContext context(parms);
    bfv_performance_test(context);
 // }
}

#endif