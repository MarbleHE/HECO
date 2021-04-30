#pragma once

#include <cstdint>
#include "ast_opt/utilities/seal_2.3.0/randomgen.h"

namespace seal_old
{
    namespace util
    {
        class RandomToStandardAdapter
        {
        public:
            typedef std::uint32_t result_type;

            RandomToStandardAdapter() : generator_(nullptr)
            {
            }

            RandomToStandardAdapter(UniformRandomGenerator *generator) : generator_(generator)
            {
            }

            const UniformRandomGenerator *generator() const
            {
                return generator_;
            }

            UniformRandomGenerator *&generator()
            {
                return generator_;
            }

            result_type operator()()
            {
                return generator_->generate();
            }

            static constexpr result_type min()
            {
                return 0;
            }

            static constexpr result_type max()
            {
                return UINT32_MAX;
            }

        private:
            UniformRandomGenerator *generator_;
        };
    }
}