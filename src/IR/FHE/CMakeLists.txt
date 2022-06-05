add_heco_dialect_library(HECOFHE
        FHEDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/FHE

        DEPENDS
        MLIRFHEIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
