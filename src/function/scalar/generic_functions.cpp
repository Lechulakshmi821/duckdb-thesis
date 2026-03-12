#include "duckdb/function/scalar/generic_functions.hpp"

namespace duckdb {

void RegisterAutoDiffGrad(BuiltinFunctions &set);
void RegisterAutoDiffGradReverse(BuiltinFunctions &set);

void BuiltinFunctions::RegisterGenericFunctions() {
        Register<ConstantOrNull>();
        Register<ExportAggregateFunction>();
        RegisterAutoDiffGrad(*this);          // forward -> mygrad_fwd
        RegisterAutoDiffGradReverse(*this);   // reverse -> mygrad_rev
}

} // namespace duckdb
