//
// Created by Moritz Winger on 05.05.21.
//

#ifndef AST_OPTIMIZER_SRC_UTILITIES_DISTRIBUTION_H_
#define AST_OPTIMIZER_SRC_UTILITIES_DISTRIBUTION_H_

class Distribution {
 private:
  double sigma = 3.2;

 public:
  Distribution(int poly_modulus_degree);
};

#endif //AST_OPTIMIZER_SRC_UTILITIES_DISTRIBUTION_H_
