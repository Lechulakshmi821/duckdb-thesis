#include "duckdb/function/scalar/generic_functions.hpp"

namespace duckdb {

void RegisterAutoDiffGrad(BuiltinFunctions &set);
//void RegisterAutoDiffGradReverse(BuiltinFunctions &set);

void BuiltinFunctions::RegisterGenericFunctions() {
	Register<ConstantOrNull>();
	Register<ExportAggregateFunction>();
        //Register<AutoDiffFun>();
        //Register<AutoDiffGradFun>();
        RegisterAutoDiffGrad(*this);
        //RegisterAutoDiffGradReverse(*this);
}

} // namespace duckdb
