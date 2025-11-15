
// mygrad.cpp
#include "duckdb/function/scalar/generic_functions.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"

namespace duckdb {

// ---------------------------------------------------------------------
// Wrapper
// ---------------------------------------------------------------------
struct AutoDiffGradFun {
	static void RegisterFunction(BuiltinFunctions &set);
};

// ---------------------------------------------------------------------
// Bind data
// ---------------------------------------------------------------------
struct AutoDiffGradBindData : public FunctionData {
	vector<LogicalType> input_types;            // N scalar inputs (exclude lambda)
	unique_ptr<Expression> bound_body;          // bound lambda body
	idx_t param_count;                           // lambda arity (fallback if branch doesn't expose it)
	vector<unique_ptr<Expression>> captures;     // bound capture expressions
	vector<LogicalType> field_types;             // struct child types [a.., da.., result]

	AutoDiffGradBindData(vector<LogicalType> input_types_p,
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
		for (auto &c : captures) cap_copies.push_back(c->Copy());
		return make_uniq<AutoDiffGradBindData>(input_types,
		                                       bound_body->Copy(),
		                                       param_count,
		                                       std::move(cap_copies),
		                                       field_types);
	}
	bool Equals(const FunctionData &) const override { return false; }
};

// ---------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------
static string AD_ParamName(idx_t i) {
	// a, b, c, ... z, aa, ab, ...
	string s;
	for (idx_t x = i;;) {
		char letter = static_cast<char>('a' + (x % 26));
		s.insert(s.begin(), letter);
		if (x < 26) break;
		x = x / 26 - 1;
	}
	return s;
}

static bool IsIntegral(const LogicalType &t) {
	switch (t.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return true;
	default:
		return false;
	}
}

static void SafeCast(ClientContext &ctx, Vector &src, Vector &dst, idx_t count) {
	VectorOperations::Cast(ctx, src, dst, count, false);
}

// out_double = cast(src_any -> DOUBLE) + delta
static void MakeDoublePlus(ClientContext &ctx, Vector &src_any, double delta, idx_t count, Vector &out_double) {
	Vector tmp(LogicalType::DOUBLE);
	SafeCast(ctx, src_any, tmp, count);

	out_double.SetVectorType(VectorType::FLAT_VECTOR);
	auto *dst = FlatVector::GetData<double>(out_double);
	ValidityMask &valid = FlatVector::Validity(out_double);
	valid.SetAllValid(count);

	UnifiedVectorFormat uvf;
	tmp.ToUnifiedFormat(count, uvf);
	auto sel = uvf.sel;
	auto data_ptr = (const double *)uvf.data;

	for (idx_t i = 0; i < count; i++) {
		const idx_t ridx = sel->get_index(i);
		double v = uvf.validity.RowIsValid(ridx) ? data_ptr[ridx] : 0.0;
		dst[i] = v + delta;
	}
}

// dst_double = (plus_double - base_double) / delta
static void FD_Derivative(const Vector &base_double, const Vector &plus_double, idx_t count, double delta, Vector &dst_double) {
	dst_double.SetVectorType(VectorType::FLAT_VECTOR);
	auto *dst = FlatVector::GetData<double>(dst_double);
	ValidityMask &valid = FlatVector::Validity(dst_double);
	valid.SetAllValid(count);

	UnifiedVectorFormat uvf_b, uvf_p;
	const_cast<Vector &>(base_double).ToUnifiedFormat(count, uvf_b);
	const_cast<Vector &>(plus_double).ToUnifiedFormat(count, uvf_p);

	auto sel_b = uvf_b.sel;
	auto sel_p = uvf_p.sel;
	auto data_b = (const double *)uvf_b.data;
	auto data_p = (const double *)uvf_p.data;

	for (idx_t i = 0; i < count; i++) {
		const idx_t bi = sel_b->get_index(i);
		const idx_t pi = sel_p->get_index(i);
		double vb = uvf_b.validity.RowIsValid(bi) ? data_b[bi] : 0.0;
		double vp = uvf_p.validity.RowIsValid(pi) ? data_p[pi] : 0.0;
		dst[i] = (vp - vb) / delta;
	}
}

// ---------------------------------------------------------------------
// Binder
// ---------------------------------------------------------------------
static unique_ptr<FunctionData> AutoDiffGradBind(ClientContext &,
                                                 ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);
	if (arguments.back()->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("mygrad: last argument must be a lambda, e.g. (a,b) -> a*a + a*b + b*b");
	}
	auto &ble = arguments.back()->Cast<BoundLambdaExpression>();

	const idx_t n_inputs = arguments.size() - 1;

	vector<LogicalType> input_types;
	input_types.reserve(n_inputs);
	for (idx_t i = 0; i < n_inputs; i++) {
		input_types.push_back(arguments[i]->return_type);
	}

	auto body = ble.lambda_expr->Copy();

	// Use branch value if available; keep a safe fallback.
	idx_t param_count = 2;
	// param_count = ble.parameter_count;

	// Copy captures (we WILL evaluate them at runtime)
	vector<unique_ptr<Expression>> cap_copy;
	cap_copy.reserve(ble.captures.size());
	for (auto &c : ble.captures) cap_copy.push_back(c->Copy());

	// STRUCT fields: a.. (input types), da.. (DOUBLE), result (body type)
	std::vector<std::pair<std::string, LogicalType>> children;
	vector<LogicalType> field_types;
	children.reserve(param_count * 2 + 1);
	field_types.reserve(param_count * 2 + 1);

	for (idx_t i = 0; i < param_count; i++) {
		const idx_t src = (n_inputs == 0) ? 0 : (i < n_inputs ? i : (n_inputs - 1));
		const auto ftype = (n_inputs == 0) ? LogicalType::SQLNULL : input_types[src];
		children.emplace_back(AD_ParamName(i), ftype);
		field_types.push_back(ftype);
	}
	for (idx_t i = 0; i < param_count; i++) {
		children.emplace_back(std::string("d") + AD_ParamName(i), LogicalType::DOUBLE);
		field_types.push_back(LogicalType::DOUBLE);
	}
	children.emplace_back("result", body->return_type);
	field_types.push_back(body->return_type);

	bound_function.return_type = LogicalType::STRUCT(std::move(children));

	return make_uniq<AutoDiffGradBindData>(std::move(input_types),
	                                       std::move(body),
	                                       param_count,
	                                       std::move(cap_copy),
	                                       std::move(field_types));
}

// ---------------------------------------------------------------------
// Executor (finite differences) with CAPTURES SUPPORT
// ---------------------------------------------------------------------
static void AutoDiffGradExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<AutoDiffGradBindData>();

	ClientContext &ctx = state.GetContext();
	const idx_t nrows  = args.size();
	const idx_t n_in   = args.ColumnCount();
	const idx_t pcount = bind.param_count > 0 ? bind.param_count : 1;

	// 1) Build param_chunk of target types (cast if needed; reuse last input if lambda expects more)
	vector<LogicalType> ptypes;
	ptypes.reserve(pcount);
	for (idx_t i = 0; i < pcount; i++) {
		const idx_t src_t = bind.input_types.empty() ? 0 : (i < bind.input_types.size() ? i : (bind.input_types.size() - 1));
		ptypes.push_back(bind.input_types.empty() ? LogicalType::SQLNULL : bind.input_types[src_t]);
	}
	DataChunk param_chunk;
	param_chunk.Initialize(Allocator::DefaultAllocator(), ptypes);
	param_chunk.SetCardinality(nrows);
	for (idx_t i = 0; i < pcount; i++) {
		if (n_in == 0) {
			param_chunk.data[i].SetVectorType(VectorType::CONSTANT_VECTOR);
			param_chunk.data[i].Reference(Value()); // NULLs
		} else {
			const idx_t src = (i < n_in ? i : (n_in - 1));
			if (args.data[src].GetType() == param_chunk.data[i].GetType()) {
				param_chunk.data[i].Reference(args.data[src]);
			} else {
				SafeCast(ctx, args.data[src], param_chunk.data[i], nrows);
			}
		}
	}

	// 2) Evaluate captures ONCE into captures_chunk
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

	// 3) Build the lambda input chunk = [params..., captures...]
	vector<LogicalType> ltypes = ptypes;
	for (auto &t : cap_types) ltypes.push_back(t);

	DataChunk lambda_chunk;
	lambda_chunk.Initialize(Allocator::DefaultAllocator(), ltypes);
	lambda_chunk.SetCardinality(nrows);

	// copy params
	for (idx_t i = 0; i < pcount; i++) {
		lambda_chunk.data[i].Reference(param_chunk.data[i]);
	}
	// append captures
	for (idx_t i = 0; i < cap_types.size(); i++) {
		lambda_chunk.data[pcount + i].Reference(captures_chunk.data[i]);
	}

	// 4) Evaluate base f(x)
	DataChunk f_base;
	f_base.Initialize(Allocator::DefaultAllocator(), {bind.bound_body->return_type});
	f_base.SetCardinality(nrows);
	{
		ExpressionExecutor exec(ctx, *bind.bound_body);
		exec.Execute(lambda_chunk, f_base);
	}
	Vector f_base_double(LogicalType::DOUBLE);
	SafeCast(ctx, f_base.data[0], f_base_double, nrows);

	// 5) Prepare final_chunk [a.., da.., result]
	const idx_t total_fields = pcount + pcount + 1;
	DataChunk final_chunk;
	final_chunk.Initialize(Allocator::DefaultAllocator(), bind.field_types);
	final_chunk.SetCardinality(nrows);

	// inputs
	for (idx_t i = 0; i < pcount; i++) {
		final_chunk.data[i].Reference(param_chunk.data[i]);
	}

	// 6) Finite-difference for each parameter (swap in lambda_chunk)
	for (idx_t k = 0; k < pcount; k++) {
		const LogicalType &t = param_chunk.data[k].GetType();
		const double delta = IsIntegral(t) ? 1.0 : 1e-6;

		Vector param_k_plus(t);
		if (IsIntegral(t)) {
			Vector src_plus_double(LogicalType::DOUBLE);
			MakeDoublePlus(ctx, param_chunk.data[k], 1.0, nrows, src_plus_double);
			SafeCast(ctx, src_plus_double, param_k_plus, nrows);
		} else {
			Vector src_plus_double(LogicalType::DOUBLE);
			MakeDoublePlus(ctx, param_chunk.data[k], delta, nrows, src_plus_double);
			if (t == LogicalType::DOUBLE) {
				param_k_plus.Reference(src_plus_double);
			} else {
				SafeCast(ctx, src_plus_double, param_k_plus, nrows);
			}
		}

		// swap in lambda_chunk
		Vector backup_k(t);
		backup_k.Reference(lambda_chunk.data[k]);
		lambda_chunk.data[k].Reference(param_k_plus);

		// f(x + delta e_k)
		DataChunk f_plus;
		f_plus.Initialize(Allocator::DefaultAllocator(), {bind.bound_body->return_type});
		f_plus.SetCardinality(nrows);
		{
			ExpressionExecutor exec(ctx, *bind.bound_body);
			exec.Execute(lambda_chunk, f_plus);
		}

		// restore
		lambda_chunk.data[k].Reference(backup_k);

		// derivative
		Vector f_plus_double(LogicalType::DOUBLE);
		SafeCast(ctx, f_plus.data[0], f_plus_double, nrows);
		FD_Derivative(f_base_double, f_plus_double, nrows, delta, final_chunk.data[pcount + k]);
	}

	// 7) result value
	final_chunk.data[pcount + pcount].Reference(f_base.data[0]);

	// 8) Pack into STRUCT
	auto &children = StructVector::GetEntries(result);
	if (children.size() < total_fields) {
		for (idx_t i = children.size(); i < total_fields; i++) {
			children.push_back(make_uniq<Vector>(bind.field_types[i]));
		}
	} else if (children.size() > total_fields) {
		children.resize(total_fields);
	}
	for (idx_t i = 0; i < total_fields; i++) {
		if (children[i]->GetType() != bind.field_types[i]) {
			children[i] = make_uniq<Vector>(bind.field_types[i]);
		}
		children[i]->Reference(final_chunk.data[i]);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------
static void AddOverloadGrad(ScalarFunctionSet &set, idx_t n_inputs) {
	vector<LogicalType> types;
	types.reserve(n_inputs + 1);
	for (idx_t i = 0; i < n_inputs; i++) types.push_back(LogicalType::ANY);
	types.push_back(LogicalType::LAMBDA);

	auto fun = ScalarFunction(types, LogicalType::ANY, AutoDiffGradExecute, AutoDiffGradBind);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

void AutoDiffGradFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet fs("mygrad");
	AddOverloadGrad(fs, 1);
	AddOverloadGrad(fs, 2);
	AddOverloadGrad(fs, 3);
	set.AddFunction(fs);
}

struct AutoDiffGradFun {
    static void RegisterFunction(BuiltinFunctions &set) {
        
    }
};

} // namespace duckdb

