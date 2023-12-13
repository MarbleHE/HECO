set(LLVM_TARGET_DEFINITIONS InsertRelinearize.td)
mlir_tablegen(InsertRelinearize.cpp.inc -gen-rewriters)
add_public_tablegen_target(evarelinearize)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOInsertRelinearize
        InsertRelinearize.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        evarelinearize

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOEVADialect
        )

