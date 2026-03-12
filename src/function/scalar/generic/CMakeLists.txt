add_library_unity(duckdb_func_generic_main OBJECT
    constant_or_null.cpp
)

add_library_unity(duckdb_func_generic_autodiff_fwd OBJECT
    autodiff_grad.cpp
)

add_library_unity(duckdb_func_generic_autodiff_rev OBJECT
    autodiff_reverse.cpp
)

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES}
    $<TARGET_OBJECTS:duckdb_func_generic_main>
    $<TARGET_OBJECTS:duckdb_func_generic_autodiff_fwd>
    $<TARGET_OBJECTS:duckdb_func_generic_autodiff_rev>
    PARENT_SCOPE)
