add_mlir_dialect_library(MLIRABC
        ABCDialect.cpp
        ABCOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ABC

        DEPENDS
        MLIRABCOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
