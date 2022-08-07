configure_lit_site_cfg(
        ${CMAKE_CURRENT_LIST_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_LIST_DIR}/lit.cfg.py
)

set(TOOLCHAIN_TEST_DEPENDS
        FileCheck count not
        fhe-tool
        )

add_lit_testsuite(check-heco "Running Tests for MLIR Toolchain"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TOOLCHAIN_TEST_DEPENDS}
        )
set_target_properties(check-heco PROPERTIES FOLDER "Tests")

add_lit_testsuites(TOOLCHAIN ${CMAKE_CURRENT_LIST_DIR} DEPENDS ${TOOLCHAIN_TEST_DEPENDS})