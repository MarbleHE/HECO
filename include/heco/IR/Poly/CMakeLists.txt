add_heco_dialect(Poly poly)
# add_heco_dialect(dialect dialect_namespace)
# This tells tablegen to look for ${dialect}.td and generate the following C++ files from it:
# ${dialect}.h.inc (Operation Declarations), ${dialect}.cpp.inc (Operation Definitions)
# ${dialect}Types.h.inc (Type Declarations), ${dialect}Types.cpp.inc (Type Definitions)
# ${dialect}Dialect.h.inc (Dialect Declarations), ${dialect}Dialect.cpp.inc (Dialect Definitions)
# and adds the target MLIR${dialect}IncGen to do this tabelgen to C++ conversion

add_heco_doc(Poly PolyDialect Poly/ -gen-dialect-doc)
# add_heco_doc(doc_filename output_file output_directory command)
# This tells CMAKE to look at ${doc_filename}.td and generate ${output_file}.md in ${output_directory}
# using the ${command} flags when doing the tablegen.
# This gets added as a dependency to mlir-doc
# Here, we're calling it for the overall Dialect Doc

add_heco_doc(Poly PolyAttrs Poly/ -gen-attrdef-doc -dialect=Poly)
# Same as above, but for attribute definitions

add_heco_doc(Poly PolyOps Poly/ -gen-op-doc -dialect=Poly)
# Same as above, but for operation definitions

add_heco_doc(Poly PolyTypes Poly/ -gen-typedef-doc -dialect=Poly)
# Same as above, but for type definitions