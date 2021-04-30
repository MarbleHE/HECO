#include <random>
#include <chrono>
#include "ast_opt/utilities/seal_2.3.0/randomgen.h"

using namespace std;

namespace seal_old
{
    UniformRandomGeneratorFactory *UniformRandomGeneratorFactory::default_factory_ = new StandardRandomAdapterFactory<random_device>();
}
