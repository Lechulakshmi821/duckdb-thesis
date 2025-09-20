// src/function/scalar/generic/mygrad_ad.cpp
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"

#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"

#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/expression_class.hpp"
#include "duckdb/common/string_util.hpp"

#include <cmath>

namespace duckdb {

// ==================== Dual number type & ops ====================
struct AD_Dual { double v; double d; };
static inline AD_Dual AD_Const(double c) { return {c, 0.0}; }
static inline AD_Dual AD_Add(AD_Dual a, AD_Dual b) { return {a.v + b.v, a.d + b.d}; }
static inline AD_Dual AD_Sub(AD_Dual a, AD_Dual b) { return {a.v - b.v, a.d - b.d}; }
static inline AD_Dual AD_Mul(AD_Dual a, AD_Dual b) { return {a.v * b.v, a.v * b.d + b.v * a.d}; }
static inline AD_Dual AD_Div(AD_Dual a, AD_Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
static inline AD_Dual AD_Log(AD_Dual x) { return {std::log(x.v), x.d / x.v}; }
static inline AD_Dual AD_Pow(AD_Dual x, AD_Dual y) {
	double v = std::pow(x.v, y.v);
	double d = v * (y.d * std::log(x.v) + y.v * (x.d / x.v));
	return {v, d};
}
static inline AD_Dual AD_LogBase(AD_Dual x, AD_Dual b) { return AD_Div(AD_Log(x), AD_Log(b)); }

// ==================== helpers ====================
static string AD_ParamName(idx_t i) {
	string s; idx_t x = i;
	while (true) {
		char c = char('a' + (x % 26));
		s.insert(s.begin(), c);
		if (x < 26) break;
		x = x / 26 - 1;
	}
	return s;
}
static double AD_ValueToDouble(const Value &val) {
	if (val.IsNull()) return 0.0;
	Value v = val;
	if (v.type() != LogicalType::DOUBLE) v = v.DefaultCastAs(LogicalType::DOUBLE);
	return v.GetValue<double>();
}

// ==================== bind data ====================
struct ADGradBindData : public FunctionData {
	vector<LogicalType> input_types;
	unique_ptr<Expression> body;
	idx_t param_count;
	vector<unique_ptr<Expression>> captures;
	vector<LogicalType> field_types;

	ADGradBindData(vector<LogicalType> in, unique_ptr<Expression> body_p, idx_t pc,
	               vector<unique_ptr<Expression>> caps, vector<LogicalType> ftypes)
	    : input_types(std::move(in)), body(std::move(body_p)), param_count(pc),
	      captures(std::move(caps)), field_types(std::move(ftypes)) {}

	unique_ptr<FunctionData> Copy() const override {
		vector<unique_ptr<Expression>> cap_copies;
		cap_copies.reserve(captures.size());
		for (auto &c : captures) cap_copies.push_back(c->Copy());
		return make_uniq<ADGradBindData>(input_types, body->Copy(), param_count, std::move(cap_copies), field_types);
	}
	bool Equals(const FunctionData &) const override { return false; }
};

// ==================== binder ====================
static unique_ptr<FunctionData> ADGradBind(ClientContext &, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &args) {
	if (args.empty() || args.back()->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("mygrad_ad: last argument must be a lambda, e.g. (a,b) -> a*a + a*b + b*b");
	}
	auto &ble = args.back()->Cast<BoundLambdaExpression>();

	const idx_t n_inputs = args.size() - 1;

	vector<LogicalType> input_types;
	input_types.reserve(n_inputs);
	for (idx_t i = 0; i < n_inputs; i++) input_types.push_back(args[i]->return_type);

	auto body = ble.lambda_expr->Copy();

	// If your branch exposes ble.parameter_count, use it; otherwise fallback:
	idx_t param_count = 2;
	// param_count = ble.parameter_count;

	vector<unique_ptr<Expression>> cap_copy;
	cap_copy.reserve(ble.captures.size());
	for (auto &c : ble.captures) cap_copy.push_back(c->Copy());

	// Output struct schema: [a.., da.., result]
	std::vector<std::pair<std::string, LogicalType>> children;
	vector<LogicalType> ftypes;
	children.reserve(param_count * 2 + 1);
	ftypes.reserve(param_count * 2 + 1);

	for (idx_t i = 0; i < param_count; i++) {
		LogicalType t = (n_inputs == 0) ? LogicalType::SQLNULL
		                                : (i < n_inputs ? input_types[i] : input_types.back());
		children.emplace_back(AD_ParamName(i), t);
		ftypes.push_back(t);
	}
	for (idx_t i = 0; i < param_count; i++) {
		children.emplace_back("d" + AD_ParamName(i), LogicalType::DOUBLE);
		ftypes.push_back(LogicalType::DOUBLE);
	}
	children.emplace_back("result", body->return_type);
	ftypes.push_back(body->return_type);

	bound_function.return_type = LogicalType::STRUCT(std::move(children));
	return make_uniq<ADGradBindData>(std::move(input_types), std::move(body), param_count, std::move(cap_copy), std::move(ftypes));
}

// ==================== tiny evaluator over bound expressions ====================
struct AD_EvalEnv {
	DataChunk *lambda_chunk; // [params..., captures...]
	idx_t pcount;
	idx_t seed_k;
	idx_t row;
	double GetAsDouble(idx_t j) const { return AD_ValueToDouble(lambda_chunk->data[j].GetValue(row)); }
};

static AD_Dual AD_EvalDual(const Expression &e, const AD_EvalEnv &env);

static inline AD_Dual AD_EvalChild(const BoundFunctionExpression &bf, const AD_EvalEnv &env, idx_t i) {
	return AD_EvalDual(*bf.children[i], env);
}
static inline AD_Dual AD_EvalChild(const BoundOperatorExpression &bo, const AD_EvalEnv &env, idx_t i) {
	return AD_EvalDual(*bo.children[i], env);
}

static AD_Dual AD_EvalDual(const Expression &e, const AD_EvalEnv &env) {
	switch (e.expression_class) {
	case ExpressionClass::BOUND_CONSTANT: {
		auto &bc = e.Cast<BoundConstantExpression>();
		return AD_Const(AD_ValueToDouble(bc.value));
	}
	case ExpressionClass::BOUND_REFERENCE: {
		auto &br = e.Cast<BoundReferenceExpression>();
		double x = env.GetAsDouble(br.index);
		double d = (br.index < env.pcount && br.index == env.seed_k) ? 1.0 : 0.0;
		return {x, d};
	}
	case ExpressionClass::BOUND_CAST: {
		auto &bc = e.Cast<BoundCastExpression>();
		return AD_EvalDual(*bc.child, env); // keep derivative through numeric casts
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &bo = e.Cast<BoundOperatorExpression>();
		if (bo.children.size() == 1) { // unary +/- 
			auto x = AD_EvalDual(*bo.children[0], env);
			if (bo.type == ExpressionType::OPERATOR_MINUS) return AD_Sub(AD_Const(0.0), x);
			return x; // unary plus
		}
		auto lhs = AD_EvalDual(*bo.children[0], env);
		auto rhs = AD_EvalDual(*bo.children[1], env);
		switch (bo.type) {
		case ExpressionType::OPERATOR_ADD:      return AD_Add(lhs, rhs);
		case ExpressionType::OPERATOR_SUBTRACT: return AD_Sub(lhs, rhs);
		case ExpressionType::OPERATOR_MULTIPLY: return AD_Mul(lhs, rhs);
		case ExpressionType::OPERATOR_DIVIDE:   return AD_Div(lhs, rhs);
		case ExpressionType::OPERATOR_POWER:    return AD_Pow(lhs, rhs);
		default:
			throw NotImplementedException("mygrad_ad: unsupported operator");
		}
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &bf = e.Cast<BoundFunctionExpression>();
		string name = StringUtil::Lower(bf.function.name);
		if (name == "add")        return AD_Add(AD_EvalChild(bf, env, 0), AD_EvalChild(bf, env, 1));
		if (name == "subtract")   return AD_Sub(AD_EvalChild(bf, env, 0), AD_EvalChild(bf, env, 1));
		if (name == "multiply")   return AD_Mul(AD_EvalChild(bf, env, 0), AD_EvalChild(bf, env, 1));
		if (name == "divide")     return AD_Div(AD_EvalChild(bf, env, 0), AD_EvalChild(bf, env, 1));
		if (name == "pow" || name == "power")
			return AD_Pow(AD_EvalChild(bf, env, 0), AD_EvalChild(bf, env, 1));
		if (name == "ln" || name == "log") {
			if (bf.children.size() == 1) return AD_Log(AD_EvalChild(bf, env, 0));
			return AD_LogBase(AD_EvalChild(bf, env, 0), AD_EvalChild(bf, env, 1));
		}
		if (name == "log10") return AD_LogBase(AD_EvalChild(bf, env, 0), AD_Const(10.0));
		throw NotImplementedException("mygrad_ad: unsupported function in lambda: " + name);
	}
	default:
		throw NotImplementedException("mygrad_ad: unsupported expression class");
	}
}

// ==================== execute ====================
static void MyGradADExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<ADGradBindData>();

	ClientContext &ctx = state.GetContext();
	const idx_t nrows  = args.size();
	const idx_t n_in   = args.ColumnCount();
	const idx_t pcount = bind.param_count > 0 ? bind.param_count : 1;

	// Build param chunk
	vector<LogicalType> ptypes;
	ptypes.reserve(pcount);
	for (idx_t i = 0; i < pcount; i++) {
		if (bind.input_types.empty()) {
			ptypes.push_back(LogicalType::SQLNULL);
		} else {
			ptypes.push_back(i < bind.input_types.size() ? bind.input_types[i] : bind.input_types.back());
		}
	}
	DataChunk param_chunk;
	param_chunk.Initialize(Allocator::DefaultAllocator(), ptypes);
	param_chunk.SetCardinality(nrows);
	for (idx_t i = 0; i < pcount; i++) {
		if (n_in == 0) {
			param_chunk.data[i].SetVectorType(VectorType::CONSTANT_VECTOR);
			param_chunk.data[i].Reference(Value());
		} else {
			const idx_t src = (i < n_in ? i : (n_in - 1));
			if (args.data[src].GetType() == param_chunk.data[i].GetType()) {
				param_chunk.data[i].Reference(args.data[src]);
			} else {
				VectorOperations::Cast(ctx, args.data[src], param_chunk.data[i], nrows, false);
			}
		}
	}

	// captures -> chunk
	vector<LogicalType> cap_types;
	cap_types.reserve(bind.captures.size());
	for (auto &c : bind.captures) cap_types.push_back(c->return_type);

	DataChunk captures_chunk;
	if (!cap_types.empty()) {
		captures_chunk.Initialize(Allocator::DefaultAllocator(), cap_types);
		captures_chunk.SetCardinality(nrows);
		for (idx_t i = 0; i < bind.captures.size(); i++) {
			ExpressionExecutor cap_exec(ctx, *bind.captures[i]);
			DataChunk cap_out;
			cap_out.Initialize(Allocator::DefaultAllocator(), {bind.captures[i]->return_type});
			cap_out.SetCardinality(nrows);
			cap_exec.Execute(args, cap_out);
			captures_chunk.data[i].Reference(cap_out.data[0]);
		}
	}

	// lambda input = [params..., captures...]
	vector<LogicalType> ltypes = ptypes;
	for (auto &t : cap_types) ltypes.push_back(t);
	DataChunk lambda_chunk;
	lambda_chunk.Initialize(Allocator::DefaultAllocator(), ltypes);
	lambda_chunk.SetCardinality(nrows);
	for (idx_t i = 0; i < pcount; i++) lambda_chunk.data[i].Reference(param_chunk.data[i]);
	for (idx_t i = 0; i < cap_types.size(); i++) lambda_chunk.data[pcount + i].Reference(captures_chunk.data[i]);

	// base value using DuckDB executor
	DataChunk f_base;
	f_base.Initialize(Allocator::DefaultAllocator(), {bind.body->return_type});
	f_base.SetCardinality(nrows);
	{
		ExpressionExecutor exec(ctx, *bind.body);
		exec.Execute(lambda_chunk, f_base);
	}

	// output: [a.., da.., result]
	const idx_t total_fields = pcount + pcount + 1;
	DataChunk final_chunk;
	final_chunk.Initialize(Allocator::DefaultAllocator(), bind.field_types);
	final_chunk.SetCardinality(nrows);

	for (idx_t i = 0; i < pcount; i++) final_chunk.data[i].Reference(param_chunk.data[i]);
	for (idx_t k = 0; k < pcount; k++) {
		final_chunk.data[pcount + k].SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::Validity(final_chunk.data[pcount + k]).SetAllValid(nrows);
	}

	for (idx_t r = 0; r < nrows; r++) {
		for (idx_t k = 0; k < pcount; k++) {
			AD_EvalEnv env{&lambda_chunk, pcount, k, r};
			AD_Dual out = AD_EvalDual(*bind.body, env);
			auto *dst = FlatVector::GetData<double>(final_chunk.data[pcount + k]);
			dst[r] = out.d;
		}
	}

	final_chunk.data[pcount + pcount].Reference(f_base.data[0]);

	// pack into STRUCT vector
	auto &children = StructVector::GetEntries(result);
	if (children.size() != total_fields) {
		children.clear();
		for (idx_t i = 0; i < total_fields; i++) {
			children.push_back(make_uniq<Vector>(bind.field_types[i]));
		}
	}
	for (idx_t i = 0; i < total_fields; i++) {
		children[i]->Reference(final_chunk.data[i]);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// ==================== registration helpers ====================
static void AD_AddOverload(ScalarFunctionSet &set, idx_t n_inputs) {
	vector<LogicalType> types;
	types.reserve(n_inputs + 1);
	for (idx_t i = 0; i < n_inputs; i++) types.push_back(LogicalType::ANY);
	types.push_back(LogicalType::LAMBDA);

	auto fun = ScalarFunction(types, LogicalType::ANY, MyGradADExecute, ADGradBind);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

struct AutoDiffADFun {
	static void RegisterFunction(BuiltinFunctions &set) {
		ScalarFunctionSet fs("mygrad_ad");
		AD_AddOverload(fs, 1);
		AD_AddOverload(fs, 2);
		AD_AddOverload(fs, 3);
		set.AddFunction(fs);
	}
};

// Public registrar expected by generic_functions.cpp
void RegisterMyGradAD(BuiltinFunctions &set) {
	AutoDiffADFun::RegisterFunction(set);
}

} // namespace duckdb
