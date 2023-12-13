set(LLVM_TARGET_DEFINITIONS LazyModswitch.td)
mlir_tablegen(LazyModswitch.cpp.inc -gen-rewriters)
add_public_tablegen_target(evalazymodswitch)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLazyModswitch
        LazyModswitch.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        evalazymodswitch

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOEVADialect
        )

