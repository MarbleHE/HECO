//
// Created by Moritz Winger on 05.05.21.
//

#ifndef AST_OPTIMIZER_SRC_UTILITIES_DISTRIBUTION_H_
#define AST_OPTIMIZER_SRC_UTILITIES_DISTRIBUTION_H_

class Distribution {
 private:
  double _sigma = 3.2;
  int _poly_modulus_degree = 0;
  int _coeff_modulus = 0;


 public:
  Distribution(int poly_modulus_degree, int number_samples, int coeff_modulus);
};

#endif //AST_OPTIMIZER_SRC_UTILITIES_DISTRIBUTION_H_
