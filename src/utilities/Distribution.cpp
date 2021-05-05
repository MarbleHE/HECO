//
// Created by Moritz Winger on 05.05.21.
//

#include "ast_opt/utilities/Distribution.h"

// create a histogram for each coefficient
Distribution::Distribution(int poly_modulus_degree, int number_samples, int coeff_modulus) {
  // want: distribution around 0:  [−B, B], where B is the bound: B = 6 * sigma (SEAL) in R_q (q = coeff_modulus)
  //       for each coeff a_i  (i \in {0,...,n-1})
  int mean = 0;
  double sigma = this->_sigma;


  //std::normal_distribution<double> distribution(mean,sigma);


}
