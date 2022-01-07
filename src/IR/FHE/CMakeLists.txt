add_mlir_dialect_library(MLIRFHE
        FHEDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/FHE

        DEPENDS
        MLIRFHEOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
