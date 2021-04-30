#pragma once

#include <cstdint>
#include "ast_opt/utilities/seal_2.3.0/smallmodulus.h"
#include "ast_opt/utilities/seal_2.3.0/util/mempool.h"
#include "ast_opt/utilities/seal_2.3.0/util/smallntt.h"
#include "ast_opt/utilities/seal_2.3.0/util/polymodulus.h"

namespace seal_old
{
    namespace util
    {
        void ntt_multiply_poly_poly(const std::uint64_t *operand1, const std::uint64_t *operand2, const SmallNTTTables &tables, std::uint64_t *result, MemoryPool &pool);

        void ntt_multiply_poly_nttpoly(const std::uint64_t *operand1, const std::uint64_t *operand2, const SmallNTTTables &tables, std::uint64_t *result, MemoryPool &pool);

        void ntt_double_multiply_poly_nttpoly(const std::uint64_t *operand1, const std::uint64_t *operand2, const std::uint64_t *operand3, const SmallNTTTables &tables, std::uint64_t *result1, std::uint64_t *result2, MemoryPool &pool);

        void ntt_dot_product_bigpolyarray_nttbigpolyarray(const std::uint64_t *array1, const std::uint64_t *array2, int count, const SmallNTTTables &tables, std::uint64_t *result, MemoryPool &pool);

        void ntt_double_dot_product_bigpolyarray_nttbigpolyarrays(const std::uint64_t *array1, const std::uint64_t *array2, const std::uint64_t *array3, int count, const SmallNTTTables &tables, std::uint64_t *result1, std::uint64_t *result2, MemoryPool &pool);

        void nussbaumer_multiply_poly_poly_coeffmod(const std::uint64_t *operand1, const std::uint64_t *operand2, int coeff_count_power, const SmallModulus &modulus, std::uint64_t *result, MemoryPool &pool);

        void nussbaumer_dot_product_bigpolyarray_coeffmod(const std::uint64_t *array1, const std::uint64_t *array2, int count, const PolyModulus &poly_modulus, const SmallModulus &modulus, std::uint64_t *result, MemoryPool &pool);
    }
}