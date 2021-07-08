#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_PERFORMANCESEAL_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_PERFORMANCESEAL_H_

#include <seal/seal.h>
#include <chrono>
#include <vector>
#include <fstream>

void bfv_performance_test(seal::SEALContext context);

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const seal::SEALContext &context)
{
  auto &context_data = *context.key_context_data();

  /*
  Which scheme are we using?
  */
  std::string scheme_name;
  switch (context_data.parms().scheme())
  {
    case seal::scheme_type::bfv:
      scheme_name = "BFV";
      break;
    case seal::scheme_type::ckks:
      scheme_name = "CKKS";
      break;
    default:
      throw std::invalid_argument("unsupported scheme");
  }
  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters :" << std::endl;
  std::cout << "|   scheme: " << scheme_name << std::endl;
  std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

  /*
  Print the size of the true (product) coefficient modulus.
  */
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_modulus_size = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
  {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;

  /*
   * Print values of coeff mulduli q_i
   */
  std::cout << "|   coeff_modulus q (q_1, ..., q_k): ";
  std::cout << context_data.parms().coeff_modulus().data()->value() << " (";
  for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
  {
    std::cout << coeff_modulus[i].value() << ", ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ")" << std::endl;

  /*
  For the BFV scheme print the plain_modulus parameter.
  */
  if (context_data.parms().scheme() == seal::scheme_type::bfv)
  {
    std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
  }

  std::cout << "\\" << std::endl;
}

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_PERFORMANCESEAL_H_
