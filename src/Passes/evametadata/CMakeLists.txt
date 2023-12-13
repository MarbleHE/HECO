set(LLVM_TARGET_DEFINITIONS MarkMetadata.td)
mlir_tablegen(MarkMetadata.cpp.inc -gen-rewriters)
add_public_tablegen_target(evametadata)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOMarkMetadata
        MarkMetadata.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        evametadata

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOEVADialect
        )

