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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <vector>

namespace duckdb {

// Tile size for vectorized reverse-mode tape replay.
// Chosen so that the per-tile working set (TILE_SIZE * prog.size() doubles
// for each of vals and adj) fits in L2 cache.
static constexpr idx_t MYGRAD_REV_TILE_SIZE = 256;

//------------------------------------------------------------------------------
// RowInput
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
// NameToSlot (your current logic)
//------------------------------------------------------------------------------
static inline bool NameToSlot(const string &raw_in, idx_t param_cnt, idx_t &slot_out) {
	string s = raw_in;
	StringUtil::Trim(s);
	s = StringUtil::Lower(s);

	// Special-case 4-arg lambdas: (w,b,x,y)
	if (param_cnt == 4) {
		if (s == "w") { slot_out = 0; return true; }
		if (s == "b") { slot_out = 1; return true; }
		if (s == "x") { slot_out = 2; return true; }
		if (s == "y") { slot_out = 3; return true; }
	}

	// Backward-compatible mapping for <= 3 params
	if (param_cnt <= 3) {
		if (s == "u" || s == "x" || s == "a") { slot_out = 0; return slot_out < param_cnt; }
		if (s == "v" || s == "y" || s == "b") { slot_out = 1; return slot_out < param_cnt; }
		if (s == "w" || s == "z" || s == "c") { slot_out = 2; return slot_out < param_cnt; }
	}

	// Fallback: names with digits like p1, x14, etc.
	idx_t pos = 0;
	while (pos < s.size() && !std::isdigit((unsigned char)s[pos])) {
		pos++;
	}
	if (pos == s.size()) return false;

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
// Compiled reverse-mode "bytecode"
//------------------------------------------------------------------------------
enum class OpKind : uint8_t { INPUT, CONST, ADD, SUB, MUL, DIV, NEG };

struct CompiledOp {
	OpKind op;
	int32_t a = -1;          // left
	int32_t b = -1;          // right
	int32_t input_slot = -1; // INPUT only
	double cval = 0.0;       // CONST only
};

//------------------------------------------------------------------------------
// Bind data (RENAMED to avoid layout/ODR issues)
//------------------------------------------------------------------------------
struct AutoDiffGradCompiledBindData : public FunctionData {
	unique_ptr<Expression> lambda_expr;
	idx_t param_cnt;

	std::vector<CompiledOp> prog;
	int32_t root = -1;

	// slot -> node index in prog (or -1 if unused)
	std::vector<int32_t> input_node;

	AutoDiffGradCompiledBindData(unique_ptr<Expression> lambda_expr_p, idx_t param_cnt_p)
	    : lambda_expr(std::move(lambda_expr_p)), param_cnt(param_cnt_p), input_node(param_cnt_p, -1) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto out = make_uniq<AutoDiffGradCompiledBindData>(lambda_expr ? lambda_expr->Copy() : nullptr, param_cnt);
		out->prog = prog;
		out->root = root;
		out->input_node = input_node;
		return out;
	}

	bool Equals(const FunctionData &) const override {
		return true;
	}
};

//------------------------------------------------------------------------------
// Emit helpers
//------------------------------------------------------------------------------
static inline int32_t EmitInput(std::vector<CompiledOp> &prog, int32_t slot) {
	CompiledOp o;
	o.op = OpKind::INPUT;
	o.input_slot = slot;
	prog.push_back(o);
	return (int32_t)prog.size() - 1;
}

static inline int32_t EmitConst(std::vector<CompiledOp> &prog, double v) {
	CompiledOp o;
	o.op = OpKind::CONST;
	o.cval = v;
	prog.push_back(o);
	return (int32_t)prog.size() - 1;
}

static inline int32_t EmitNeg(std::vector<CompiledOp> &prog, int32_t a) {
	CompiledOp o;
	o.op = OpKind::NEG;
	o.a = a;
	prog.push_back(o);
	return (int32_t)prog.size() - 1;
}

static inline int32_t EmitBin(std::vector<CompiledOp> &prog, OpKind k, int32_t a, int32_t b) {
	CompiledOp o;
	o.op = k;
	o.a = a;
	o.b = b;
	prog.push_back(o);
	return (int32_t)prog.size() - 1;
}

//------------------------------------------------------------------------------
// Compiler: Expression -> prog
//------------------------------------------------------------------------------
static int32_t CompileExpr(Expression &expr,
                           idx_t param_cnt,
                           std::vector<CompiledOp> &prog,
                           std::vector<int32_t> &input_cache,
                           std::vector<int32_t> &input_node_out) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_CONSTANT: {
		auto &c = expr.Cast<BoundConstantExpression>();
		return EmitConst(prog, c.value.GetValue<double>());
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &c = expr.Cast<BoundColumnRefExpression>();
		idx_t slot = c.binding.column_index;
		if (slot >= param_cnt) return EmitConst(prog, 0.0);

		if (input_cache[slot] < 0) {
			input_cache[slot] = EmitInput(prog, (int32_t)slot);
			input_node_out[slot] = input_cache[slot];
		}
		return input_cache[slot];
	}
	case ExpressionClass::BOUND_REF: {
		string nm = expr.alias;
		if (nm.empty()) nm = expr.ToString();

		idx_t slot;
		if (!NameToSlot(nm, param_cnt, slot)) {
			return EmitConst(prog, 0.0);
		}
		if (input_cache[slot] < 0) {
			input_cache[slot] = EmitInput(prog, (int32_t)slot);
			input_node_out[slot] = input_cache[slot];
		}
		return input_cache[slot];
	}
	case ExpressionClass::BOUND_CAST: {
		auto &c = expr.Cast<BoundCastExpression>();
		return CompileExpr(*c.child, param_cnt, prog, input_cache, input_node_out);
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		auto &ch = op.children;
		const string t = StringUtil::Lower(ExpressionTypeToString(op.type));

		if (ch.size() == 1 && (t.find("neg") != string::npos || t.find("unary") != string::npos || t == "-")) {
			auto a = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out);
			return EmitNeg(prog, a);
		}
		if (ch.size() != 2) {
			throw BinderException("mygrad_rev: only unary minus and binary +,-,*,/ supported");
		}

		auto L = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out);
		auto R = CompileExpr(*ch[1], param_cnt, prog, input_cache, input_node_out);

		if (t.find("add") != string::npos || t.find("plus") != string::npos) return EmitBin(prog, OpKind::ADD, L, R);
		if (t.find("sub") != string::npos || t.find("minus") != string::npos) return EmitBin(prog, OpKind::SUB, L, R);
		if (t.find("mul") != string::npos) return EmitBin(prog, OpKind::MUL, L, R);
		if (t.find("div") != string::npos) return EmitBin(prog, OpKind::DIV, L, R);

		throw BinderException("mygrad_rev: unsupported operator");
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		auto &ch = fn.children;
		const string name = StringUtil::Lower(fn.function.name);

		if (ch.size() == 1 && (name.find("neg") != string::npos || name == "-")) {
			auto a = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out);
			return EmitNeg(prog, a);
		}
		if (ch.size() == 2) {
			auto L = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out);
			auto R = CompileExpr(*ch[1], param_cnt, prog, input_cache, input_node_out);

			if (name == "+" || name.find("add") != string::npos || name.find("plus") != string::npos) return EmitBin(prog, OpKind::ADD, L, R);
			if (name == "-" || name.find("sub") != string::npos || name.find("minus") != string::npos) return EmitBin(prog, OpKind::SUB, L, R);
			if (name == "*" || name.find("mul") != string::npos) return EmitBin(prog, OpKind::MUL, L, R);
			if (name == "/" || name.find("div") != string::npos) return EmitBin(prog, OpKind::DIV, L, R);
		}
		throw BinderException("mygrad_rev: unsupported function in compiler");
	}
	default:
		throw BinderException("mygrad_rev: unsupported expression type in compiler");
	}
}

//------------------------------------------------------------------------------
// Bind: compile once
//------------------------------------------------------------------------------
static unique_ptr<FunctionData>
AutoDiffGradBind(ClientContext &, ScalarFunction &bound_function, vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < 2) {
		throw BinderException("mygrad_rev: expected at least (x..., lambda)");
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
		return make_uniq<AutoDiffGradCompiledBindData>(std::move(dummy_lambda), n_numeric_args);
	}

	auto &ble = last_arg->Cast<BoundLambdaExpression>();
	const idx_t n_params = ble.parameter_count;

	if (n_params == 0) throw BinderException("mygrad_rev: lambda must have at least 1 parameter");
	if (n_params != n_numeric_args) throw BinderException("mygrad_rev: lambda parameter count must match #numeric args");
	if (!ble.captures.empty()) throw BinderException("mygrad_rev: captures not supported");

	auto lambda_expr = ble.lambda_expr->Copy();

	for (idx_t i = 0; i < n_numeric_args; i++) {
		if (arguments[i]->return_type.id() != LogicalTypeId::DOUBLE) {
			throw BinderException("mygrad_rev: numeric args must be DOUBLE");
		}
	}

	auto bind_data = make_uniq<AutoDiffGradCompiledBindData>(std::move(lambda_expr), n_params);

	bind_data->prog.clear();
	bind_data->prog.reserve(256);
	std::fill(bind_data->input_node.begin(), bind_data->input_node.end(), -1);

	std::vector<int32_t> input_cache(n_params, -1);
	bind_data->root = CompileExpr(*bind_data->lambda_expr, n_params, bind_data->prog, input_cache, bind_data->input_node);

	return std::move(bind_data);
}

//------------------------------------------------------------------------------
// Execute: run compiled program per row
//------------------------------------------------------------------------------
static void AutoDiffGradExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<AutoDiffGradCompiledBindData>();

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

	const auto &prog = bind.prog;
	const int32_t root = bind.root;

	// Tiled vectorized tape replay.
	// Working buffers are sized per tile so they fit in L2 cache.
	// Within each tile, the per-row loop runs the original forward/backward
	// passes; only the indexing is local to the tile.
	const idx_t TILE = MYGRAD_REV_TILE_SIZE;
	std::vector<double> vals(prog.size() * TILE);
	std::vector<double> adj(prog.size() * TILE);

	for (idx_t tile_start = 0; tile_start < count; tile_start += TILE) {
		const idx_t tile_count = std::min(TILE, count - tile_start);

		// Phase A: forward pass - for each operation, process all rows in this tile
		for (idx_t i = 0; i < prog.size(); i++) {
			const auto &op = prog[i];
			switch (op.op) {
			case OpKind::INPUT:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = in.Get((idx_t)op.input_slot, tile_start + r);
				break;
			case OpKind::CONST:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = op.cval;
				break;
			case OpKind::NEG:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = -vals[(idx_t)op.a * TILE + r];
				break;
			case OpKind::ADD:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = vals[(idx_t)op.a * TILE + r] + vals[(idx_t)op.b * TILE + r];
				break;
			case OpKind::SUB:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = vals[(idx_t)op.a * TILE + r] - vals[(idx_t)op.b * TILE + r];
				break;
			case OpKind::MUL:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = vals[(idx_t)op.a * TILE + r] * vals[(idx_t)op.b * TILE + r];
				break;
			case OpKind::DIV:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = vals[(idx_t)op.a * TILE + r] / vals[(idx_t)op.b * TILE + r];
				break;
			default:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = 0.0;
				break;
			}
		}

		// Phase B: zero adjoints for this tile, seed root for every row in tile
		for (idx_t i = 0; i < prog.size(); i++) {
			for (idx_t r = 0; r < tile_count; r++) adj[i * TILE + r] = 0.0;
		}
		if (root >= 0) {
			for (idx_t r = 0; r < tile_count; r++) adj[(idx_t)root * TILE + r] = 1.0;
		}

		// Phase C: backward pass - for each operation in reverse, process all rows in tile
		for (idx_t ii = prog.size(); ii > 0; ii--) {
			const idx_t i = ii - 1;
			const auto &op = prog[i];
			switch (op.op) {
			case OpKind::INPUT:
			case OpKind::CONST:
				break;
			case OpKind::NEG:
				for (idx_t r = 0; r < tile_count; r++) adj[(idx_t)op.a * TILE + r] += adj[i * TILE + r] * (-1.0);
				break;
			case OpKind::ADD:
				for (idx_t r = 0; r < tile_count; r++) {
					const double a = adj[i * TILE + r];
					adj[(idx_t)op.a * TILE + r] += a;
					adj[(idx_t)op.b * TILE + r] += a;
				}
				break;
			case OpKind::SUB:
				for (idx_t r = 0; r < tile_count; r++) {
					const double a = adj[i * TILE + r];
					adj[(idx_t)op.a * TILE + r] += a;
					adj[(idx_t)op.b * TILE + r] += a * (-1.0);
				}
				break;
			case OpKind::MUL:
				for (idx_t r = 0; r < tile_count; r++) {
					const double a = adj[i * TILE + r];
					const double lv = vals[(idx_t)op.a * TILE + r];
					const double rv = vals[(idx_t)op.b * TILE + r];
					adj[(idx_t)op.a * TILE + r] += a * rv;
					adj[(idx_t)op.b * TILE + r] += a * lv;
				}
				break;
			case OpKind::DIV:
				for (idx_t r = 0; r < tile_count; r++) {
					const double a = adj[i * TILE + r];
					const double lv = vals[(idx_t)op.a * TILE + r];
					const double rv = vals[(idx_t)op.b * TILE + r];
					const double inv = 1.0 / rv;
					adj[(idx_t)op.a * TILE + r] += a * inv;
					adj[(idx_t)op.b * TILE + r] += a * (-lv) * inv * inv;
				}
				break;
			default:
				break;
			}
		}

		// Phase D: emit gradients for all rows in tile
		for (idx_t j = 0; j < N; j++) {
			const int32_t node = bind.input_node[j];
			if (node >= 0) {
				for (idx_t r = 0; r < tile_count; r++) child_ptrs[j][tile_start + r] = adj[(idx_t)node * TILE + r];
			} else {
				for (idx_t r = 0; r < tile_count; r++) child_ptrs[j][tile_start + r] = 0.0;
			}
		}
	}
}

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------
void RegisterAutoDiffGradReverse(BuiltinFunctions &set) {
	Printer::Print("[mygrad] REGISTER (reverse AD compiled, Any...,Lambda) -> STRUCT d1..dN");

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
