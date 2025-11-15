#include "duckdb/function/scalar/generic_functions.hpp"

namespace duckdb {

void RegisterMyGradLambda(BuiltinFunctions &set);

/*struct AutoDiffFun {
    static void RegisterFunction(BuiltinFunctions &set);
};*/
/*struct AutoDiffGradFun {
    static void RegisterFunction(BuiltinFunctions &set);
};*/

void BuiltinFunctions::RegisterGenericFunctions() {
	Register<ConstantOrNull>();
	Register<ExportAggregateFunction>();
        //Register<AutoDiffFun>();
        //Register<AutoDiffGradFun>();
        RegisterMyGradLambda(*this);
       
}

} // namespace duckdb
