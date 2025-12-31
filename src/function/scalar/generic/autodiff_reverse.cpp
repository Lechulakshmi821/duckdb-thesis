#include "duckdb/function/scalar/generic_functions.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/value.hpp"

#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_lambda_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <cctype>
#include <vector>
#include <limits>

namespace duckdb {

//------------------------------------------------------------------------------
// Bind data (same as your forward-mode file)
//------------------------------------------------------------------------------
struct AutoDiffGradBindData : public FunctionData {
	unique_ptr<Expression> lambda_expr;
	idx_t param_cnt;

	AutoDiffGradBindData(unique_ptr<Expression> lambda_expr_p, idx_t param_cnt_p)
	    : lambda_expr(std::move(lambda_expr_p)), param_cnt(param_cnt_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AutoDiffGradBindData>(lambda_expr ? lambda_expr->Copy() : nullptr, param_cnt);
	}

	bool Equals(const FunctionData &) const override {
		return true;
	}
};

//------------------------------------------------------------------------------
// RowInput (same idea as your forward-mode file)
//------------------------------------------------------------------------------
struct RowInput {
	idx_t n;
	vector<const double *> ptrs;

	explicit RowInput(DataChunk &args, idx_t n_p) : n(n_p) {
		ptrs.resize(n);
		for (idx_t c = 0; c < n; c++) {
			args.data[c].Flatten(args.size());
			ptrs[c] = FlatVector::GetData<double>(args.data[c]);
		}
	}

	inline double Get(idx_t col, idx_t row) const {
		D_ASSERT(col < n);
		return ptrs[col][row];
	}
};

//------------------------------------------------------------------------------
// Name-to-slot (copied from your forward-mode file; keep same behavior)
//------------------------------------------------------------------------------
static inline bool NameToSlot(const string &raw_in, idx_t param_cnt, idx_t &slot_out) {
	string s = raw_in;
	StringUtil::Trim(s);
	s = StringUtil::Lower(s);

	if (s == "u" || s == "x" || s == "a") { slot_out = 0; return slot_out < param_cnt; }
	if (s == "v" || s == "y" || s == "b") { slot_out = 1; return slot_out < param_cnt; }
	if (s == "w" || s == "z" || s == "c") { slot_out = 2; return slot_out < param_cnt; }

	idx_t pos = 0;
	while (pos < s.size() && !std::isdigit((unsigned char)s[pos])) {
		pos++;
	}
	if (pos == s.size()) {
		return false;
	}

	uint64_t num = 0;
	for (idx_t i = pos; i < s.size(); i++) {
		if (!std::isdigit((unsigned char)s[i])) break;
		num = num * 10 + (uint64_t)(s[i] - '0');
	}
	if (num == 0) return false;
	slot_out = (idx_t)(num - 1);
	return slot_out < param_cnt;
}

//------------------------------------------------------------------------------
// Reverse-mode tape
//------------------------------------------------------------------------------
// Each node stores its value and how it depends on up to 2 parents:
//
// For z = f(x,y):
//   dz/dx_local = d0
//   dz/dy_local = d1
//
// Backprop:
//   adj[x] += adj[z] * d0
//   adj[y] += adj[z] * d1
//
struct TapeNode {
	double val = 0.0;

	// parents (children in the computational graph)
	idx_t p0 = DConstants::INVALID_INDEX;
	idx_t p1 = DConstants::INVALID_INDEX;

	// local partials
	double d0 = 0.0;
	double d1 = 0.0;

	// if this node is a reused leaf for an input slot (0..N-1), store it (optional)
	idx_t input_slot = DConstants::INVALID_INDEX;
};

struct Tape {
	vector<TapeNode> nodes;

	inline idx_t AddConst(double v) {
		TapeNode n;
		n.val = v;
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	inline idx_t AddInput(double v, idx_t slot) {
		TapeNode n;
		n.val = v;
		n.input_slot = slot;
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	inline idx_t AddUnaryNeg(idx_t a) {
		TapeNode n;
		n.val = -nodes[a].val;
		n.p0 = a;
		n.d0 = -1.0;
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	inline idx_t AddAdd(idx_t a, idx_t b) {
		TapeNode n;
		n.val = nodes[a].val + nodes[b].val;
		n.p0 = a; n.p1 = b;
		n.d0 = 1.0; n.d1 = 1.0;
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	inline idx_t AddSub(idx_t a, idx_t b) {
		TapeNode n;
		n.val = nodes[a].val - nodes[b].val;
		n.p0 = a; n.p1 = b;
		n.d0 = 1.0; n.d1 = -1.0;
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	inline idx_t AddMul(idx_t a, idx_t b) {
		const double av = nodes[a].val;
		const double bv = nodes[b].val;
		TapeNode n;
		n.val = av * bv;
		n.p0 = a; n.p1 = b;
		n.d0 = bv; // d(av*bv)/dav
		n.d1 = av; // d(av*bv)/dbv
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	inline idx_t AddDiv(idx_t a, idx_t b) {
		const double av = nodes[a].val;
		const double bv = nodes[b].val;
		TapeNode n;
		n.val = av / bv;
		n.p0 = a; n.p1 = b;

		// d(av/bv)/dav = 1/bv
		// d(av/bv)/dbv = -av/(bv*bv)
		const double inv = 1.0 / bv;
		n.d0 = inv;
		n.d1 = -av * inv * inv;

		nodes.push_back(n);
		return nodes.size() - 1;
	}
};

//------------------------------------------------------------------------------
// Reverse-mode expression eval: build tape, return root node index
// Also caches input leaves so repeated references reuse the same node.
//------------------------------------------------------------------------------
static idx_t EvalExprReverse(Expression &expr,
                            const RowInput &in,
                            idx_t row,
                            const AutoDiffGradBindData &bind,
                            Tape &tape,
                            vector<idx_t> &input_leaf_cache) {
	const idx_t N = bind.param_cnt;

	switch (expr.expression_class) {
	case ExpressionClass::BOUND_CONSTANT: {
		auto &c = expr.Cast<BoundConstantExpression>();
		return tape.AddConst(c.value.GetValue<double>());
	}

	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &c = expr.Cast<BoundColumnRefExpression>();
		const idx_t slot = c.binding.column_index;
		if (slot >= N) {
			return tape.AddConst(0.0);
		}
		if (input_leaf_cache[slot] == DConstants::INVALID_INDEX) {
			input_leaf_cache[slot] = tape.AddInput(in.Get(slot, row), slot);
		}
		return input_leaf_cache[slot];
	}

	case ExpressionClass::BOUND_REF: {
		string nm = expr.alias;
		if (nm.empty()) nm = expr.ToString();

		idx_t slot;
		if (!NameToSlot(nm, N, slot)) {
			return tape.AddConst(0.0);
		}
		if (input_leaf_cache[slot] == DConstants::INVALID_INDEX) {
			input_leaf_cache[slot] = tape.AddInput(in.Get(slot, row), slot);
		}
		return input_leaf_cache[slot];
	}

	case ExpressionClass::BOUND_CAST: {
		auto &c = expr.Cast<BoundCastExpression>();
		return EvalExprReverse(*c.child, in, row, bind, tape, input_leaf_cache);
	}

	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		auto &ch = op.children;
		const string t = StringUtil::Lower(ExpressionTypeToString(op.type));

		if (ch.size() == 1 && (t.find("neg") != string::npos || t.find("unary") != string::npos || t == "-")) {
			auto a = EvalExprReverse(*ch[0], in, row, bind, tape, input_leaf_cache);
			return tape.AddUnaryNeg(a);
		}
		if (ch.size() != 2) {
			throw BinderException("mygrad: only unary minus and binary +,-,*,/ supported");
		}

		auto L = EvalExprReverse(*ch[0], in, row, bind, tape, input_leaf_cache);
		auto R = EvalExprReverse(*ch[1], in, row, bind, tape, input_leaf_cache);

		if (t.find("add") != string::npos || t.find("plus") != string::npos) return tape.AddAdd(L, R);
		if (t.find("sub") != string::npos || t.find("minus") != string::npos) return tape.AddSub(L, R);
		if (t.find("mul") != string::npos) return tape.AddMul(L, R);
		if (t.find("div") != string::npos) return tape.AddDiv(L, R);

		throw BinderException("mygrad: unsupported operator (use +,-,*,/)");
	}

	case ExpressionClass::BOUND_FUNCTION: {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		auto &ch = fn.children;
		const string name = StringUtil::Lower(fn.function.name);

		if (ch.size() == 1 && (name.find("neg") != string::npos || name == "-")) {
			auto a = EvalExprReverse(*ch[0], in, row, bind, tape, input_leaf_cache);
			return tape.AddUnaryNeg(a);
		}

		if (ch.size() == 2) {
			auto L = EvalExprReverse(*ch[0], in, row, bind, tape, input_leaf_cache);
			auto R = EvalExprReverse(*ch[1], in, row, bind, tape, input_leaf_cache);

			if (name == "+" || name.find("add") != string::npos || name.find("plus") != string::npos) return tape.AddAdd(L, R);
			if (name == "-" || name.find("sub") != string::npos || name.find("minus") != string::npos) return tape.AddSub(L, R);
			if (name == "*" || name.find("mul") != string::npos) return tape.AddMul(L, R);
			if (name == "/" || name.find("div") != string::npos) return tape.AddDiv(L, R);
		}

		throw BinderException("mygrad: unsupported function in reverse AD");
	}

	default:
		throw BinderException("mygrad: unsupported expression type in reverse AD");
	}
}

//------------------------------------------------------------------------------
// Bind (same logic as your forward-mode file)
//------------------------------------------------------------------------------
static unique_ptr<FunctionData>
AutoDiffGradBind(ClientContext &, ScalarFunction &bound_function, vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < 2) {
		throw BinderException("mygrad: expected at least (x..., lambda)");
	}

	const idx_t n_numeric_args = arguments.size() - 1;

	child_list_t<LogicalType> kids;
	kids.reserve(n_numeric_args);
	for (idx_t i = 0; i < n_numeric_args; i++) {
		kids.push_back({string("d") + std::to_string(i + 1), LogicalType::DOUBLE});
	}
	bound_function.return_type = LogicalType::STRUCT(std::move(kids));

	auto *last_arg = arguments.back().get();

	if (last_arg->expression_class != ExpressionClass::BOUND_LAMBDA) {
		auto dummy_lambda = make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0));
		return make_uniq<AutoDiffGradBindData>(std::move(dummy_lambda), n_numeric_args);
	}

	auto &ble = last_arg->Cast<BoundLambdaExpression>();
	const idx_t n_params = ble.parameter_count;

	if (n_params == 0) throw BinderException("mygrad: lambda must have at least 1 parameter");
	if (n_params != n_numeric_args) throw BinderException("mygrad: lambda parameter count must match #numeric args");

	if (!ble.captures.empty()) {
		throw BinderException("mygrad: captures not supported. Pass constants as explicit parameters");
	}

	auto lambda_expr = ble.lambda_expr->Copy();

	for (idx_t i = 0; i < n_numeric_args; i++) {
		if (arguments[i]->return_type.id() != LogicalTypeId::DOUBLE) {
			throw BinderException("mygrad: numeric args must be DOUBLE (use ::DOUBLE)");
		}
	}

	return make_uniq<AutoDiffGradBindData>(std::move(lambda_expr), n_params);
}

//------------------------------------------------------------------------------
// Execute (reverse mode): build tape per row, backprop once, write d1..dN
//------------------------------------------------------------------------------
static void AutoDiffGradExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<AutoDiffGradBindData>();

	const idx_t N = bind.param_cnt;
	const idx_t count = args.size();
	if (count == 0) return;

	RowInput in(args, N);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (idx_t j = 0; j < N; j++) {
		children[j]->SetVectorType(VectorType::FLAT_VECTOR);
	}

	vector<double *> child_ptrs(N);
	for (idx_t j = 0; j < N; j++) {
		child_ptrs[j] = FlatVector::GetData<double>(*children[j]);
	}

	for (idx_t r = 0; r < count; r++) {
		Tape tape;
		tape.nodes.reserve(64);

		// cache leaf nodes for each input slot so repeated references reuse the same node
		vector<idx_t> input_leaf_cache(N, DConstants::INVALID_INDEX);

		const idx_t root = EvalExprReverse(*bind.lambda_expr, in, r, bind, tape, input_leaf_cache);

		// backprop adjoints
		vector<double> adj(tape.nodes.size(), 0.0);
		adj[root] = 1.0;

		for (idx_t i = tape.nodes.size(); i > 0; i--) {
			const idx_t idx = i - 1;
			const auto &n = tape.nodes[idx];
			const double a = adj[idx];

			if (n.p0 != DConstants::INVALID_INDEX) {
				adj[n.p0] += a * n.d0;
			}
			if (n.p1 != DConstants::INVALID_INDEX) {
				adj[n.p1] += a * n.d1;
			}
		}

		// emit gradients for each input slot (if unused, gradient remains 0)
		for (idx_t j = 0; j < N; j++) {
			double g = 0.0;
			const idx_t leaf = input_leaf_cache[j];
			if (leaf != DConstants::INVALID_INDEX) {
				g = adj[leaf];
			}
			child_ptrs[j][r] = g;
		}
	}
}

//------------------------------------------------------------------------------
// Register (same outward signature; now reverse-mode implementation)
//------------------------------------------------------------------------------
void RegisterAutoDiffGrad(BuiltinFunctions &set) {
	Printer::Print("[mygrad] REGISTER (reverse AD, Any...,Lambda; NAME-based seeding; N-variable) -> STRUCT d1..dN");

	ScalarFunctionSet fset("mygrad_rev");
	for (idx_t n = 1; n <= 32; n++) {
		vector<LogicalType> args;
		for (idx_t i = 0; i < n; i++) args.push_back(LogicalType::ANY);
		args.push_back(LogicalType::LAMBDA);

		ScalarFunction fn(std::move(args), LogicalType::ANY, AutoDiffGradExecute, AutoDiffGradBind);
		fn.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		fset.AddFunction(std::move(fn));
	}
	set.AddFunction(fset);
}

} // namespace duckdb
