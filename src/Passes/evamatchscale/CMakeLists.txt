set(LLVM_TARGET_DEFINITIONS MatchScale.td)
mlir_tablegen(MatchScale.cpp.inc -gen-rewriters)
add_public_tablegen_target(evamatchscale)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOMatchScale
        MatchScale.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        evamatchscale

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOEVADialect
        )

