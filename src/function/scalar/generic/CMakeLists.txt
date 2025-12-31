add_library_unity(duckdb_func_generic_main OBJECT constant_or_null.cpp
    #autodiff.cpp
    #autodiff_grad.cpp
    #mygrad_lambda.cpp
    autodiff_grad.cpp
    #autodiff_reverse.cpp
)

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_func_generic_main>
    PARENT_SCOPE)
