#include "duckdb/function/scalar/generic_functions.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"

namespace duckdb {

struct AutoDiffLambdaBindData : public FunctionData {
	LogicalType stype;
	unique_ptr<Expression> lambda_expr;

	AutoDiffLambdaBindData(const LogicalType &stype_p, unique_ptr<Expression> lambda_expr) 
    : stype(stype_p), lambda_expr(std::move(lambda_expr)) {

	};
	~AutoDiffLambdaBindData() override {};

public:
	bool Equals(const FunctionData &other_p) const override {
    	auto &other = other_p.Cast<AutoDiffLambdaBindData>();
    	return lambda_expr->Equals(other.lambda_expr.get()) && stype == other.stype;
	};
	unique_ptr<FunctionData> Copy() const override { return make_uniq<AutoDiffLambdaBindData>(stype, lambda_expr->Copy()); };
	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
		throw NotImplementedException("FIXME: lambda serialize");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            ScalarFunction &bound_function) {
		throw NotImplementedException("FIXME: lambda deserialize");
	}
};

static unique_ptr<FunctionData> AutoDiffBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	// at least a column and the lambda function
	D_ASSERT(arguments.size() == 2);
	if (arguments[1]->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("Invalid lambda expression!");
	}
	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	//bound_function.return_type = arguments[0]->return_type;
	auto lambda_expr = std::move(bound_lambda_expr.lambda_expr);
	return make_uniq<AutoDiffLambdaBindData>(bound_function.return_type,std::move(lambda_expr));
}


static void AutoDiffFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// get the lambda expression
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<AutoDiffLambdaBindData>();
	auto &lambda_expr = info.lambda_expr;
	DataChunk lambda_chunk;
	lambda_chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType(LogicalType::BIGINT)});
	ExpressionExecutor expr_executor(state.GetContext(), *lambda_expr);
	expr_executor.Execute(args,lambda_chunk);
	auto &lambda_vector = lambda_chunk.data[0];
	result.Reference(lambda_vector);
}

void AutoDiffFun::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("myautodiff", {LogicalType::BIGINT, LogicalType::LAMBDA}, LogicalType::BIGINT, AutoDiffFunction, AutoDiffBind); //, nullptr, nullptr);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize =   AutoDiffLambdaBindData::Serialize;
	fun.deserialize = AutoDiffLambdaBindData::Deserialize;
	set.AddFunction(fun);
}

} // namespace duckdb
