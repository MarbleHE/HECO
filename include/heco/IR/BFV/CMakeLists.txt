add_heco_dialect(BFV bfv) 
# add_heco_dialect(dialect dialect_namespace)
# This tells tablegen to look for ${dialect}.td and generate the following C++ files from it:
# ${dialect}.h.inc (Operation Declarations), ${dialect}.cpp.inc (Operation Definitions)
# ${dialect}Types.h.inc (Type Declarations), # ${dialect}Types.cpp.inc (Type Definitions)
# ${dialect}Dialect.h.inc (Dialect Declarations), # ${dialect}Dialect.cpp.inc (Dialect Definitions)
# and adds the target MLIR${dialect}IncGen to do this tabelgen to C++ conversion

add_heco_doc(BFV BFVDialect BFV/ -gen-dialect-doc)
# add_heco_doc(doc_filename output_file output_directory command)
# This tells CMAKE to look at ${doc_filename}.td and generate ${output_file}.md in ${output_directory}
# using the ${command} flag when doing the tablegen.
# This gets added as a dependency to mlir-doc
# Here, we're calling it for the overall Dialect Doc

add_heco_doc(BFV BFVAttrs BFV/ -gen-attrdef-doc -dialect=BFV)
# Same as above, but for attribute definitions

add_heco_doc(BFV BFVOps BFV/ -gen-op-doc -dialect=BFV)
# Same as above, but for operation definitions

add_heco_doc(BFV BFVTypes BFV/ -gen-typedef-doc -dialect=BFV)
# Same as above, but for type definitions