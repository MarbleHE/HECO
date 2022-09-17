add_heco_dialect_library(HECOCKKSDialect
        CKKSDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/CKKS

        DEPENDS
        MLIRCKKSIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
