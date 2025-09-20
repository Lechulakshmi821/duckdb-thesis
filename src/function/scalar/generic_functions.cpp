#include "duckdb/function/scalar/generic_functions.hpp"

namespace duckdb {

void RegisterMyGradAD(BuiltinFunctions &set); 

void BuiltinFunctions::RegisterGenericFunctions() {
	Register<ConstantOrNull>();
	Register<ExportAggregateFunction>();
        Register<AutoDiffFun>();
        Register<AutoDiffGradFun>();
        RegisterMyGradAD(*this);   
}

} // namespace duckdb
