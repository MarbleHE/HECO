include(AddMLIRPython)

pybind11_add_module(pyabc
        abc_wrapper.cpp
)

target_compile_options(pyabc PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    # Enable RTTI and exceptions.
    -frtti -fexceptions
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    # Enable RTTI and exceptions.
    /EHsc /GR>
)

# Configure the output to match python expectations.
set_target_properties(
    pyabc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "."
    OUTPUT_NAME "_abc_wrapper"
    NO_SONAME ON
)

# XXX: Add libraries used in python wrapper here
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ast_opt
    ast_opt_mlir
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRABC
)
llvm_update_compile_flags(pyabc)
target_link_libraries(pyabc
    PRIVATE
    ${LIBS}
    ${PYEXT_LIBADD}
)

target_link_options(pyabc
    PRIVATE
    # On Linux, disable re-export of any static linked libraries that
    # came through.
    $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
)

mlir_check_all_link_libraries(pyabc)


