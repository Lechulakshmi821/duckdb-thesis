
#include "duckdb/function/scalar/generic_functions.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

// -----------------------------------------------------------------------------
// Bind data: keep lambda body, return type, arity, and CAPTURES
// -----------------------------------------------------------------------------
struct AutoDiffLambdaBindData : public FunctionData {
	LogicalType stype;                                  // lambda return type
	unique_ptr<Expression> lambda_expr;                 // lambda body
	idx_t parameter_count;                              // lambda arity
	vector<unique_ptr<Expression>> captures;            // captured expressions (e.g., +2)

	AutoDiffLambdaBindData(const LogicalType &stype_p,
	                       unique_ptr<Expression> lambda_expr_p,
	                       idx_t parameter_count_p,
	                       vector<unique_ptr<Expression>> captures_p)
	    : stype(stype_p),
	      lambda_expr(std::move(lambda_expr_p)),
	      parameter_count(parameter_count_p),
	      captures(std::move(captures_p)) {}

	unique_ptr<FunctionData> Copy() const override {
		vector<unique_ptr<Expression>> caps_copy;
		caps_copy.reserve(captures.size());
		for (auto &c : captures) {
			caps_copy.push_back(c->Copy());
		}
		return make_uniq<AutoDiffLambdaBindData>(stype, lambda_expr->Copy(), parameter_count, std::move(caps_copy));
	}

	bool Equals(const FunctionData &other_p) const override {
		const auto &o = other_p.Cast<AutoDiffLambdaBindData>();
		if (stype != o.stype || parameter_count != o.parameter_count) return false;
		if (!lambda_expr->Equals(*o.lambda_expr)) return false;
		if (captures.size() != o.captures.size()) return false;
		for (idx_t i = 0; i < captures.size(); i++) {
			if (!captures[i]->Equals(*o.captures[i])) return false;
		}
		return true;
	}
};

// -----------------------------------------------------------------------------
// Binder: enforce lambda, set return type, copy captures
// -----------------------------------------------------------------------------
static unique_ptr<FunctionData> AutoDiffBind(ClientContext &,
                                             ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);
	if (arguments.back()->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("myautodiff: last argument must be a lambda expression");
	}

	auto &bound_lambda = arguments.back()->Cast<BoundLambdaExpression>();
	auto lambda_expr   = std::move(bound_lambda.lambda_expr);

	const idx_t param_count = bound_lambda.parameter_count;
	if (param_count == 0) {
		throw BinderException("myautodiff: lambda must take at least one parameter");
	}

	// Return type matches lambda
	bound_function.return_type = lambda_expr->return_type;

	// Copy captures
	vector<unique_ptr<Expression>> captures_copy;
	captures_copy.reserve(bound_lambda.captures.size());
	for (auto &c : bound_lambda.captures) {
		captures_copy.push_back(c->Copy());
	}

	// Debug
	Printer::Print(StringUtil::Format(
	    "[myautodiff] BIND param_count=%d return=%s expr=%s (captures=%d)",
	    (int)param_count,
	    bound_function.return_type.ToString().c_str(),
	    lambda_expr->ToString().c_str(),
	    (int)captures_copy.size()));

	return make_uniq<AutoDiffLambdaBindData>(
	    bound_function.return_type,
	    std::move(lambda_expr),
	    param_count,
	    std::move(captures_copy));
}

// -----------------------------------------------------------------------------
// Executor: build input = [params..., captures...]; keep capture chunks alive
// -----------------------------------------------------------------------------
static void AutoDiffFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr   = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_info   = func_expr.bind_info->Cast<AutoDiffLambdaBindData>();
	auto &lambda_expr = bind_info.lambda_expr;

	const idx_t n_inputs    = args.ColumnCount();
	const idx_t param_cnt   = bind_info.parameter_count;
	const idx_t capture_cnt = bind_info.captures.size();

	D_ASSERT(n_inputs >= 1);
	D_ASSERT(param_cnt >= 1);

	Printer::Print(StringUtil::Format("myautodiff debug: n_inputs=%d param_cnt=%d",
	                                  (int)n_inputs, (int)param_cnt));

	// Types: params mirror source arg types; then capture types
	vector<LogicalType> ltypes;
	ltypes.reserve(param_cnt + capture_cnt);

	vector<idx_t> use_idx(param_cnt);
	for (idx_t i = 0; i < param_cnt; i++) {
		use_idx[i] = (i < n_inputs) ? i : (n_inputs - 1);
		ltypes.push_back(args.data[use_idx[i]].GetType());
	}
	for (auto &cap : bind_info.captures) {
		ltypes.push_back(cap->return_type);
	}

	// Prepare input chunk
	DataChunk lambda_input;
	lambda_input.Initialize(Allocator::DefaultAllocator(), ltypes);
	lambda_input.SetCardinality(args.size());

	// Reference param vectors
	for (idx_t i = 0; i < param_cnt; i++) {
		lambda_input.data[i].Reference(args.data[use_idx[i]]);
	}

	// Materialize captures and keep them alive
	vector<unique_ptr<DataChunk>> capture_chunks;
	capture_chunks.reserve(capture_cnt);
	for (idx_t c = 0; c < capture_cnt; c++) {
		auto &cap_expr = *bind_info.captures[c];

		auto cap_chunk = make_uniq<DataChunk>();
		cap_chunk->Initialize(Allocator::DefaultAllocator(), {cap_expr.return_type});
		cap_chunk->SetCardinality(args.size());

		DataChunk empty_in; // captures don't need our param columns
		empty_in.SetCardinality(args.size());

		ExpressionExecutor cap_exec(state.GetContext(), cap_expr);
		cap_exec.Execute(empty_in, *cap_chunk);

		lambda_input.data[param_cnt + c].Reference(cap_chunk->data[0]);
		capture_chunks.push_back(std::move(cap_chunk)); // keep alive until after execute
	}

	// Execute lambda
	DataChunk lambda_out;
	lambda_out.Initialize(Allocator::DefaultAllocator(), {bind_info.stype});

	Printer::Print(StringUtil::Format("[myautodiff] lambda_input cols=%d; out=%s",
	                                  (int)lambda_input.ColumnCount(),
	                                  bind_info.stype.ToString().c_str()));

	ExpressionExecutor exec(state.GetContext(), *lambda_expr);
	exec.Execute(lambda_input, lambda_out);

	result.Reference(lambda_out.data[0]);
}

// -----------------------------------------------------------------------------
// Registrar: BIGINT inputs; return type is set at bind time
// -----------------------------------------------------------------------------
void AutoDiffFun::RegisterFunction(BuiltinFunctions &set) {
	Printer::Print("[myautodiff] REGISTER myautodiff");
	ScalarFunctionSet fset("myautodiff");

	{
		// (BIGINT, LAMBDA) -> <lambda return type>
		ScalarFunction f({LogicalType::BIGINT, LogicalType::LAMBDA},
		                 LogicalType::BIGINT /* placeholder, overwritten in bind */,
		                 AutoDiffFunction, AutoDiffBind);
		f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		fset.AddFunction(std::move(f));
	}
	{
		// (BIGINT, BIGINT, LAMBDA) -> <lambda return type>
		ScalarFunction f({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::LAMBDA},
		                 LogicalType::BIGINT /* placeholder, overwritten in bind */,
		                 AutoDiffFunction, AutoDiffBind);
		f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		fset.AddFunction(std::move(f));
	}
	{
		// (BIGINT, BIGINT, BIGINT, LAMBDA) -> <lambda return type>
		ScalarFunction f({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::LAMBDA},
		                 LogicalType::BIGINT /* placeholder, overwritten in bind */,
		                 AutoDiffFunction, AutoDiffBind);
		f.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		fset.AddFunction(std::move(f));
	}

	set.AddFunction(fset);
}

} // namespace duckdb

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

// =====================================================================
// Bind data
// =====================================================================
struct AutoDiffLambdaBindData : public FunctionData {
	vector<LogicalType> input_types;           // scalar inputs (exclude lambda)
	unique_ptr<Expression> bound_body;         // bound lambda body
	idx_t param_count;                          // lambda arity (fallback if branch doesn't expose it)
	vector<unique_ptr<Expression>> captures;    // bound capture expressions
	vector<LogicalType> field_types;            // struct child types [a.., cap.., result]

	AutoDiffLambdaBindData(vector<LogicalType> input_types_p,
	                       unique_ptr<Expression> bound_body_p,
	                       idx_t param_count_p,
	                       vector<unique_ptr<Expression>> captures_p,
	                       vector<LogicalType> field_types_p)
	    : input_types(std::move(input_types_p)),
	      bound_body(std::move(bound_body_p)),
	      param_count(param_count_p),
	      captures(std::move(captures_p)),
	      field_types(std::move(field_types_p)) {}

	unique_ptr<FunctionData> Copy() const override {
		vector<unique_ptr<Expression>> cap_copies;
		cap_copies.reserve(captures.size());
		for (auto &c : captures) {
			cap_copies.push_back(c->Copy());
		}
		return make_uniq<AutoDiffLambdaBindData>(input_types,
		                                         bound_body->Copy(),
		                                         param_count,
		                                         std::move(cap_copies),
		                                         field_types);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<AutoDiffLambdaBindData>();
		if (param_count != o.param_count) return false;
		if (input_types != o.input_types) return false;
		if (field_types != o.field_types) return false;
		if (!bound_body->Equals(*o.bound_body)) return false;
		if (captures.size() != o.captures.size()) return false;
		for (idx_t i = 0; i < captures.size(); i++) {
			if (!captures[i]->Equals(*o.captures[i])) return false;
		}
		return true;
	}
};

// 0,1,2,... -> "a","b",...,"z","aa","ab",...
static string AlphabeticalName(idx_t i) {
	string s;
	idx_t x = i;
	while (true) {
		char letter = static_cast<char>('a' + (x % 26));
		s.insert(s.begin(), letter);
		if (x < 26) break;
		x = x / 26 - 1;
	}
	return s;
}

// =====================================================================
// Binder
// Signature: N scalar inputs (ANY) + last arg = BOUND_LAMBDA
// Return: STRUCT(a,b,c,..., cap1,cap2,..., result)
// =====================================================================
static unique_ptr<FunctionData> AutoDiffBind(ClientContext & /*context*/,
                                             ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);

	// Last argument must be a bound lambda
	if (arguments.back()->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("myautodiff: last argument must be a lambda, e.g. (a,b) -> a*a + a*b + b*b");
	}
	auto &ble = arguments.back()->Cast<BoundLambdaExpression>();

	// Collect input types (exclude lambda)
	const idx_t n_inputs = arguments.size() - 1;
	vector<LogicalType> input_types;
	input_types.reserve(n_inputs);
	for (idx_t i = 0; i < n_inputs; i++) {
		input_types.push_back(arguments[i]->return_type);
	}

	// Bound lambda body
	auto bound_body = ble.lambda_expr->Copy();

	// Lambda arity; use branch field if available
	idx_t param_count = 2;
	// param_count = ble.parameter_count; // enable if your tree exposes it

	// Copy captures
	vector<unique_ptr<Expression>> captures_copy;
	captures_copy.reserve(ble.captures.size());
	for (auto &c : ble.captures) {
		captures_copy.push_back(c->Copy());
	}

	// STRUCT children
	std::vector<std::pair<std::string, LogicalType>> children;
	vector<LogicalType> field_types;
	children.reserve(param_count + captures_copy.size() + 1);
	field_types.reserve(param_count + captures_copy.size() + 1);

	// Params a,b,c,... (reuse last input type if arity > inputs)
	for (idx_t i = 0; i < param_count; i++) {
		const idx_t src = (n_inputs == 0) ? 0 : (i < n_inputs ? i : (n_inputs - 1));
		const auto ftype = (n_inputs == 0) ? LogicalType::SQLNULL : input_types[src];
		children.emplace_back(AlphabeticalName(i), ftype);
		field_types.push_back(ftype);
	}

	// Captures
	for (idx_t c = 0; c < captures_copy.size(); c++) {
		children.emplace_back("cap" + to_string(c + 1), captures_copy[c]->return_type);
		field_types.push_back(captures_copy[c]->return_type);
	}

	// Result
	children.emplace_back("result", bound_body->return_type);
	field_types.push_back(bound_body->return_type);

	// Final return type
	bound_function.return_type = LogicalType::STRUCT(std::move(children));

	return make_uniq<AutoDiffLambdaBindData>(std::move(input_types),
	                                         std::move(bound_body),
	                                         param_count,
	                                         std::move(captures_copy),
	                                         std::move(field_types));
}

// =====================================================================
// Executor
// =====================================================================
static void AutoDiffExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<AutoDiffLambdaBindData>();

	const idx_t n_inputs = args.ColumnCount();
	const idx_t pcount   = bind.param_count > 0 ? bind.param_count : 1;
	const idx_t ccount   = bind.captures.size();

	// 1) Build the param chunk (a..), reusing last input if needed, CAST if types differ
	vector<LogicalType> ptypes;
	ptypes.reserve(pcount);
	for (idx_t i = 0; i < pcount; i++) {
		const idx_t src_type = bind.input_types.empty() ? 0 : (i < bind.input_types.size() ? i : (bind.input_types.size() - 1));
		ptypes.push_back(bind.input_types.empty() ? LogicalType::SQLNULL : bind.input_types[src_type]);
	}
	DataChunk param_chunk;
	param_chunk.Initialize(Allocator::DefaultAllocator(), ptypes);
	param_chunk.SetCardinality(args.size());
	for (idx_t i = 0; i < pcount; i++) {
		if (n_inputs == 0) {
			param_chunk.data[i].SetVectorType(VectorType::CONSTANT_VECTOR);
			param_chunk.data[i].Reference(Value()); // NULLs
			continue;
		}
		const idx_t src = (i < n_inputs ? i : (n_inputs - 1));
		Vector &src_vec = args.data[src];
		if (src_vec.GetType() == param_chunk.data[i].GetType()) {
			param_chunk.data[i].Reference(src_vec);
		} else {
			VectorOperations::Cast(
			    state.GetContext(),      // ClientContext&
			    src_vec,                 // source
			    param_chunk.data[i],     // dest (already has target type)
			    args.size(),             // count
			    false                    // try_strict
			);
		}
	}

	// 2) Materialize captures into columns (cap1..capM)
	vector<unique_ptr<DataChunk>> cap_chunks;
	cap_chunks.reserve(ccount);
	for (idx_t c = 0; c < ccount; c++) {
		auto &cap_expr = *bind.captures[c];
		auto cap_chunk = make_uniq<DataChunk>();
		cap_chunk->Initialize(Allocator::DefaultAllocator(), {cap_expr.return_type});
		cap_chunk->SetCardinality(args.size());

		DataChunk empty_in;
		empty_in.SetCardinality(args.size());

		ExpressionExecutor cap_exec(state.GetContext(), cap_expr);
		cap_exec.Execute(empty_in, *cap_chunk);

		cap_chunks.push_back(std::move(cap_chunk));
	}

	// 3) Evaluate bound body -> lambda_out
	DataChunk lambda_out;
	lambda_out.Initialize(Allocator::DefaultAllocator(), {bind.bound_body->return_type});
	lambda_out.SetCardinality(args.size());
	{
		ExpressionExecutor exec(state.GetContext(), *bind.bound_body);

		// Build eval_in = [a.., cap..]
		vector<LogicalType> eval_types;
		eval_types.reserve(pcount + ccount);
		for (idx_t i = 0; i < pcount; i++) eval_types.push_back(param_chunk.data[i].GetType());
		for (idx_t c = 0; c < ccount; c++) eval_types.push_back(cap_chunks[c]->data[0].GetType());

		DataChunk eval_in;
		eval_in.Initialize(Allocator::DefaultAllocator(), eval_types);
		eval_in.SetCardinality(args.size());
		for (idx_t i = 0; i < pcount; i++) {
			eval_in.data[i].Reference(param_chunk.data[i]);
		}
		for (idx_t c = 0; c < ccount; c++) {
			eval_in.data[pcount + c].Reference(cap_chunks[c]->data[0]);
		}

		exec.Execute(eval_in, lambda_out);
	}

	// 4) MATERIALIZE a final chunk with EXACT target types (bind.field_types),
	//    CASTING each source if needed. Then wire it into the STRUCT result.
	const idx_t total_fields = pcount + ccount + 1;
	DataChunk final_chunk;
	final_chunk.Initialize(Allocator::DefaultAllocator(), bind.field_types);
	final_chunk.SetCardinality(args.size());

	// helper to copy (ref or cast) into final_chunk[i] with target type bind.field_types[i]
	auto to_final = [&](idx_t out_idx, Vector &src_vec) {
		if (final_chunk.data[out_idx].GetType() == src_vec.GetType()) {
			final_chunk.data[out_idx].Reference(src_vec);
		} else {
			VectorOperations::Cast(
			    state.GetContext(),             // ClientContext&
			    src_vec,                        // source
			    final_chunk.data[out_idx],      // dest (has target type)
			    args.size(),                    // count
			    false                           // try_strict
			);
		}
	};

	// a.. params
	for (idx_t i = 0; i < pcount; i++) {
		to_final(i, param_chunk.data[i]);
	}
	// cap.. captures
	for (idx_t c = 0; c < ccount; c++) {
		to_final(pcount + c, cap_chunks[c]->data[0]);
	}
	// result
	to_final(pcount + ccount, lambda_out.data[0]);

	// Now ensure result STRUCT has children with the correct types and reference final_chunk columns
	auto &out_children = StructVector::GetEntries(result);
	if (out_children.size() < total_fields) {
		for (idx_t i = out_children.size(); i < total_fields; i++) {
			out_children.push_back(make_uniq<Vector>(bind.field_types[i]));
		}
	} else if (out_children.size() > total_fields) {
		out_children.resize(total_fields);
	}
	for (idx_t i = 0; i < total_fields; i++) {
		// force child vector to correct type for this binding
		if (out_children[i]->GetType() != bind.field_types[i]) {
			out_children[i] = make_uniq<Vector>(bind.field_types[i]);
		}
		// and now reference the final (type-correct) vector
		out_children[i]->Reference(final_chunk.data[i]);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// =====================================================================
// Registration helpers (file scope)
// =====================================================================
static void AddOverload(ScalarFunctionSet &set, idx_t n_inputs) {
	vector<LogicalType> types;
	types.reserve(n_inputs + 1);
	for (idx_t i = 0; i < n_inputs; i++) {
		types.push_back(LogicalType::ANY);
	}
	types.push_back(LogicalType::LAMBDA); // last arg = lambda

	auto fun = ScalarFunction(types, LogicalType::ANY, AutoDiffExecute, AutoDiffBind);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = nullptr;
	fun.deserialize = nullptr;
	set.AddFunction(fun);
}

// =====================================================================
// Registration entry point
// =====================================================================
void AutoDiffFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet fs("myautodiff");
	AddOverload(fs, 1); // myautodiff(col, (a)->...)
	AddOverload(fs, 2); // myautodiff(col1, col2, (a,b)->...)
	AddOverload(fs, 3); // myautodiff(col1, col2, col3, (a,b,c)->...)
	set.AddFunction(fs);
}

} // namespace duckdb
