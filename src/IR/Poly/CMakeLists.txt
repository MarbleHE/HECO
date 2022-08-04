add_heco_dialect_library(HECOPolyDialect
        PolyDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Poly

        DEPENDS
        MLIRPolyIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
