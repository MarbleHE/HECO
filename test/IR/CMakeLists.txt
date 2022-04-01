#configure_lit_site_cfg(
#        lit.site.cfg.py.in
#        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
#        MAIN_CONFIG
#        lit.cfg.py
#)
#
#set(STANDALONE_TEST_DEPENDS
#        FileCheck count not
#        abc-opt
#        abc-translate
#        )
#
#add_lit_testsuite(check-abc "Running the abc regression tests"
#        ${CMAKE_CURRENT_BINARY_DIR}
#        DEPENDS ${STANDALONE_TEST_DEPENDS}
#        )
#set_target_properties(check-abc PROPERTIES FOLDER "Tests")
#
#add_lit_testsuites(ABC FHE DEPENDS ${STANDALONE_TEST_DEPENDS})