//
// Created by Moritz Winger on 03.05.21.
//
#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include <seal/plaintext.h>

int64_t plaintext_norm(seal::Plaintext ptxt) {
  std::uint64_t* start = ptxt.data(); // pointer to first coeff of ptxt poly
  int64_t max = *start;
  for (int i = 1; i < ptxt.coeff_count(); i++) {
     if (*(start + i) > max) {
       max = *(start + i);
     }
  }
  return max;
}

#endif