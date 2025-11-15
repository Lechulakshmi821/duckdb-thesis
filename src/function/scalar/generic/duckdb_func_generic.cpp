#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/builtins.hpp"   // this file already includes it elsewhere

namespace duckdb {
ScalarFunction MakeMyGradLambda();  // from mygrad_lambda.cpp

void RegisterMyGradLambda(BuiltinFunctions &set) {
    ScalarFunctionSet fset("mygrad_lambda");
    fset.AddFunction(MakeMyGradLambda());
    set.AddFunction(fset);
}
} // namespace duckdb
