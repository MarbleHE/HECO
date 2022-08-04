add_heco_dialect_library(HECOASTDialect
        ASTDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/heco/IR/AST

        DEPENDS
        MLIRASTIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
