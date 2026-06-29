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
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>
#include <vector>

#define DUCKDB_HAVE_LLVM
#ifdef DUCKDB_HAVE_LLVM
using RevJitFuncType = void(*)(const double*, double*);
extern RevJitFuncType CompileRevJITBridge(const void* prog_ptr, const void* root_ptr, const void* input_node_ptr, uint64_t N);
#endif

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
enum class OpKind : uint8_t { INPUT, CONST, ADD, SUB, MUL, DIV, NEG, POW };

struct CompiledOp {
	OpKind op;
	int32_t a = -1;          // left
	int32_t b = -1;          // right
	int32_t input_slot = -1; // INPUT only
	double cval = 0.0;       // CONST only
};


//------------------------------------------------------------------------------
// Symbolic expression tree for bind-time derivative generation
// Following Prof. Schüle's LingoDB deriveDerivation pattern
//------------------------------------------------------------------------------
enum class SymOp : uint8_t {
    CONST,   // constant value
    PARAM,   // input parameter (slot index)
    ADD,     // left + right
    SUB,     // left - right
    MUL,     // left * right
    DIV,     // left / right
    NEG,     // -child
    POW,     // left ^ right
};

struct SymNode {
    SymOp op;
    double cval = 0.0;      // CONST only
    int32_t slot = -1;      // PARAM only
    int32_t left = -1;      // binary ops
    int32_t right = -1;     // binary ops
    int32_t child = -1;     // unary ops
};

// Symbolic expression tree — stored as a flat vector (like prog)
using SymProg = std::vector<SymNode>;

static int32_t SymConst(SymProg &s, double v) {
    SymNode n; n.op=SymOp::CONST; n.cval=v; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymParam(SymProg &s, int32_t slot) {
    SymNode n; n.op=SymOp::PARAM; n.slot=slot; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymAdd(SymProg &s, int32_t l, int32_t r) {
    SymNode n; n.op=SymOp::ADD; n.left=l; n.right=r; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymSub(SymProg &s, int32_t l, int32_t r) {
    SymNode n; n.op=SymOp::SUB; n.left=l; n.right=r; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymMul(SymProg &s, int32_t l, int32_t r) {
    SymNode n; n.op=SymOp::MUL; n.left=l; n.right=r; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymDiv(SymProg &s, int32_t l, int32_t r) {
    SymNode n; n.op=SymOp::DIV; n.left=l; n.right=r; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymNeg(SymProg &s, int32_t c) {
    SymNode n; n.op=SymOp::NEG; n.child=c; s.push_back(n);
    return (int32_t)s.size() - 1;
}
static int32_t SymPow(SymProg &s, int32_t l, int32_t r) {
    SymNode n; n.op=SymOp::POW; n.left=l; n.right=r; s.push_back(n);
    return (int32_t)s.size() - 1;
}


//------------------------------------------------------------------------------
// SymDiff: symbolic differentiation following Prof. Schüle's LingoDB pattern
// Given a compiled prog and a node index, returns the derivative node index
// with respect to param `wrt_slot`, building into `deriv` SymProg.
// `fwd` holds the forward expression nodes (mirroring prog as SymNodes).
//------------------------------------------------------------------------------
// SymForward: copy forward expression for prog[node] into deriv as self-contained SymNodes
static int32_t SymForward(const std::vector<CompiledOp> &prog, int32_t node, SymProg &deriv) {
	if (node < 0) return SymConst(deriv, 0.0);
	const auto &op = prog[(idx_t)node];
	switch (op.op) {
	case OpKind::INPUT: return SymParam(deriv, op.input_slot);
	case OpKind::CONST: return SymConst(deriv, op.cval);
	case OpKind::NEG:   return SymNeg(deriv, SymForward(prog, op.a, deriv));
	case OpKind::ADD:   return SymAdd(deriv, SymForward(prog, op.a, deriv), SymForward(prog, op.b, deriv));
	case OpKind::SUB:   return SymSub(deriv, SymForward(prog, op.a, deriv), SymForward(prog, op.b, deriv));
	case OpKind::MUL:   return SymMul(deriv, SymForward(prog, op.a, deriv), SymForward(prog, op.b, deriv));
	case OpKind::DIV:   return SymDiv(deriv, SymForward(prog, op.a, deriv), SymForward(prog, op.b, deriv));
	case OpKind::POW:   return SymPow(deriv, SymForward(prog, op.a, deriv), SymForward(prog, op.b, deriv));
	default: return SymConst(deriv, 0.0);
	}
}

static int32_t SymDiff(const std::vector<CompiledOp> &prog,
                       int32_t node,
                       int32_t wrt_slot,
                       SymProg &deriv) {
	if (node < 0) return SymConst(deriv, 0.0);
	const auto &op = prog[(idx_t)node];
	switch (op.op) {
	case OpKind::CONST:
		return SymConst(deriv, 0.0);
	case OpKind::INPUT:
		return (op.input_slot == wrt_slot) ? SymConst(deriv, 1.0) : SymConst(deriv, 0.0);
	case OpKind::NEG: {
		auto du = SymDiff(prog, op.a, wrt_slot, deriv);
		return SymNeg(deriv, du);
	}
	case OpKind::ADD: {
		auto du = SymDiff(prog, op.a, wrt_slot, deriv);
		auto dv = SymDiff(prog, op.b, wrt_slot, deriv);
		return SymAdd(deriv, du, dv);
	}
	case OpKind::SUB: {
		auto du = SymDiff(prog, op.a, wrt_slot, deriv);
		auto dv = SymDiff(prog, op.b, wrt_slot, deriv);
		return SymSub(deriv, du, dv);
	}
	case OpKind::MUL: {
		auto du = SymDiff(prog, op.a, wrt_slot, deriv);
		auto dv = SymDiff(prog, op.b, wrt_slot, deriv);
		auto u  = SymForward(prog, op.a, deriv);
		auto v  = SymForward(prog, op.b, deriv);
		auto t1 = SymMul(deriv, du, v);
		auto t2 = SymMul(deriv, u, dv);
		return SymAdd(deriv, t1, t2);
	}
	case OpKind::DIV: {
		auto du  = SymDiff(prog, op.a, wrt_slot, deriv);
		auto dv  = SymDiff(prog, op.b, wrt_slot, deriv);
		auto u   = SymForward(prog, op.a, deriv);
		auto v   = SymForward(prog, op.b, deriv);
		auto v2  = SymForward(prog, op.b, deriv);
		auto t1  = SymMul(deriv, du, v);
		auto t2  = SymMul(deriv, u, dv);
		auto num = SymSub(deriv, t1, t2);
		auto c2  = SymConst(deriv, 2.0);
		auto den = SymPow(deriv, v2, c2);
		return SymDiv(deriv, num, den);
	}
	case OpKind::POW: {
		auto du   = SymDiff(prog, op.a, wrt_slot, deriv);
		auto u    = SymForward(prog, op.a, deriv);
		auto v    = SymForward(prog, op.b, deriv);
		auto v2   = SymForward(prog, op.b, deriv);
		auto c1   = SymConst(deriv, 1.0);
		auto vm1  = SymSub(deriv, v, c1);
		auto upow = SymPow(deriv, u, vm1);
		auto t1   = SymMul(deriv, v2, upow);
		return SymMul(deriv, t1, du);
	}
	default:
		return SymConst(deriv, 0.0);
	}
}

//------------------------------------------------------------------------------
// EvalSymLinear: evaluate a SymProg node iteratively, returning a scalar value for one row
//------------------------------------------------------------------------------
// EvalSymLinear: evaluate SymProg iteratively (no recursion, pre-allocated buffer)
static double EvalSymLinear(const SymProg &prog, int32_t root,
                            const RowInput &in, idx_t row,
                            std::vector<double> &vals) {
	if (root < 0) return 0.0;
	const idx_t sz = (idx_t)root + 1;
	if (vals.size() < sz) vals.resize(sz);
	for (idx_t i = 0; i < sz; i++) {
		const auto &n = prog[i];
		switch (n.op) {
		case SymOp::CONST: vals[i] = n.cval; break;
		case SymOp::PARAM: vals[i] = in.Get((idx_t)n.slot, row); break;
		case SymOp::NEG:   vals[i] = -vals[(idx_t)n.child]; break;
		case SymOp::ADD:   vals[i] = vals[(idx_t)n.left] + vals[(idx_t)n.right]; break;
		case SymOp::SUB:   vals[i] = vals[(idx_t)n.left] - vals[(idx_t)n.right]; break;
		case SymOp::MUL:   vals[i] = vals[(idx_t)n.left] * vals[(idx_t)n.right]; break;
		case SymOp::DIV:   vals[i] = vals[(idx_t)n.left] / vals[(idx_t)n.right]; break;
		case SymOp::POW:   vals[i] = std::pow(vals[(idx_t)n.left], vals[(idx_t)n.right]); break;
		default:           vals[i] = 0.0; break;
		}
	}
	return vals[(idx_t)root];
}


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

	// Symbolic derivative trees (one SymProg per parameter)
	std::vector<SymProg> sym_derivs;
#ifdef DUCKDB_HAVE_LLVM
	void* jit_func = nullptr;
#endif
	std::vector<int32_t> sym_roots;
	std::vector<int32_t> sym_fwd;
	std::vector<double> sym_scratch; // pre-allocated eval buffer

	AutoDiffGradCompiledBindData(unique_ptr<Expression> lambda_expr_p, idx_t param_cnt_p)
	    : lambda_expr(std::move(lambda_expr_p)), param_cnt(param_cnt_p), input_node(param_cnt_p, -1) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto out = make_uniq<AutoDiffGradCompiledBindData>(lambda_expr ? lambda_expr->Copy() : nullptr, param_cnt);
		out->prog = prog;
		out->root = root;
		out->input_node = input_node;
		out->sym_derivs = sym_derivs;
		out->sym_roots = sym_roots;
		out->sym_fwd = sym_fwd;
		out->sym_scratch = sym_scratch;
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
                           std::vector<int32_t> &input_node_out,
                           const vector<unique_ptr<Expression>> &captures) {
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
		// Handle capture references like #0, #1, #2 etc.
		if (!nm.empty() && nm[0] == '#') {
			// DuckDB #N index is offset by param count; scan all captures for constants
			for (idx_t ci = 0; ci < captures.size(); ci++) {
				if (captures[ci]->expression_class == ExpressionClass::BOUND_CONSTANT) {
					auto &cc = captures[ci]->Cast<BoundConstantExpression>();
					return EmitConst(prog, cc.value.GetValue<double>());
				}
			}
			return EmitConst(prog, 0.0);
		}
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
		return CompileExpr(*c.child, param_cnt, prog, input_cache, input_node_out, captures);
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		auto &ch = op.children;
		const string t = StringUtil::Lower(ExpressionTypeToString(op.type));

		if (ch.size() == 1 && (t.find("neg") != string::npos || t.find("unary") != string::npos || t == "-")) {
			auto a = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out, captures);
			return EmitNeg(prog, a);
		}
		if (ch.size() != 2) {
			throw BinderException("mygrad_rev: only unary minus and binary +,-,*,/ supported");
		}

		auto L = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out, captures);
		auto R = CompileExpr(*ch[1], param_cnt, prog, input_cache, input_node_out, captures);

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
			auto a = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out, captures);
			return EmitNeg(prog, a);
		}
		if (ch.size() == 2) {
			auto L = CompileExpr(*ch[0], param_cnt, prog, input_cache, input_node_out, captures);
			auto R = CompileExpr(*ch[1], param_cnt, prog, input_cache, input_node_out, captures);

			if (name == "+" || name.find("add") != string::npos || name.find("plus") != string::npos) return EmitBin(prog, OpKind::ADD, L, R);
			if (name == "-" || name.find("sub") != string::npos || name.find("minus") != string::npos) return EmitBin(prog, OpKind::SUB, L, R);
			if (name == "*" || name.find("mul") != string::npos) return EmitBin(prog, OpKind::MUL, L, R);
			if (name == "/" || name.find("div") != string::npos) return EmitBin(prog, OpKind::DIV, L, R);
			if (name == "^" || name == "pow" || name.find("pow") != string::npos) return EmitBin(prog, OpKind::POW, L, R);
		}
		{ throw BinderException("mygrad_rev: unsupported function '" + name + "' (children=" + std::to_string(ch.size()) + ") in compiler"); }
	}
	case ExpressionClass::BOUND_LAMBDA: {
		// DuckDB routes captured constants through BOUND_LAMBDA
		// Try to parse the string representation as a double
		try {
			double v = std::stod(expr.ToString());
					return EmitConst(prog, v);
		} catch (...) {}
		return EmitConst(prog, 0.0);
	}
	default: {
			return EmitConst(prog, 0.0);
	}
	}
}


//------------------------------------------------------------------------------
// CSE: Common Subexpression Elimination on compiled prog
// Scans for structurally identical nodes and merges them.
// Returns a remapping vector: remap[old_idx] = new_idx
//------------------------------------------------------------------------------
static void ApplyCSE(std::vector<CompiledOp> &prog, int32_t &root) {
	if (prog.empty()) return;

	// Build canonical key for each node
	// Key: (op, a_remapped, b_remapped, input_slot, cval)
	std::vector<int32_t> remap(prog.size(), -1);
	std::vector<CompiledOp> new_prog;
	new_prog.reserve(prog.size());

	// Map from key string to new index
	std::unordered_map<std::string, int32_t> seen;

	auto make_key = [&](const CompiledOp &op, int32_t ra, int32_t rb) -> std::string {
		char buf[64];
		snprintf(buf, sizeof(buf), "%d:%d:%d:%d:%.17g",
			(int)op.op, ra, rb, op.input_slot, op.cval);
		return std::string(buf);
	};

	for (idx_t i = 0; i < prog.size(); i++) {
		CompiledOp op = prog[i];
		// Remap children
		int32_t ra = (op.a >= 0) ? remap[(idx_t)op.a] : -1;
		int32_t rb = (op.b >= 0) ? remap[(idx_t)op.b] : -1;
		op.a = ra;
		op.b = rb;

		std::string key = make_key(op, ra, rb);
		auto it = seen.find(key);
		if (it != seen.end()) {
			// Duplicate — reuse existing node
			remap[i] = it->second;
		} else {
			// New node
			int32_t new_idx = (int32_t)new_prog.size();
			new_prog.push_back(op);
			seen[key] = new_idx;
			remap[i] = new_idx;
		}
	}

	root = remap[(idx_t)root];
	prog = std::move(new_prog);
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
	for (auto &cap : ble.captures) {
			if (cap->expression_class != ExpressionClass::BOUND_CONSTANT) {
			throw BinderException("mygrad_rev: only constant captures supported, got class=" + std::to_string((int)cap->expression_class) + " str=" + cap->ToString());
		}
	}

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
	bind_data->root = CompileExpr(*bind_data->lambda_expr, n_params, bind_data->prog, input_cache, bind_data->input_node, ble.captures);

	// Apply CSE to reduce tape size
	ApplyCSE(bind_data->prog, bind_data->root);
	// Rebuild input_node mapping after CSE remap
	std::fill(bind_data->input_node.begin(), bind_data->input_node.end(), -1);
	for (idx_t i = 0; i < bind_data->prog.size(); i++) {
		if (bind_data->prog[i].op == OpKind::INPUT) {
			bind_data->input_node[(idx_t)bind_data->prog[i].input_slot] = (int32_t)i;
		}
	}

	// Build forward SymProg mirroring prog
	const idx_t prog_size = bind_data->prog.size();
	SymProg fwd_prog;
	std::vector<int32_t> fwd_idx(prog_size, -1);
	for (idx_t i = 0; i < prog_size; i++) {
		const auto &op = bind_data->prog[i];
		if      (op.op == OpKind::INPUT) fwd_idx[i] = SymParam(fwd_prog, op.input_slot);
		else if (op.op == OpKind::CONST) fwd_idx[i] = SymConst(fwd_prog, op.cval);
		else if (op.op == OpKind::NEG)   fwd_idx[i] = SymNeg(fwd_prog, fwd_idx[(idx_t)op.a]);
		else if (op.op == OpKind::ADD)   fwd_idx[i] = SymAdd(fwd_prog, fwd_idx[(idx_t)op.a], fwd_idx[(idx_t)op.b]);
		else if (op.op == OpKind::SUB)   fwd_idx[i] = SymSub(fwd_prog, fwd_idx[(idx_t)op.a], fwd_idx[(idx_t)op.b]);
		else if (op.op == OpKind::MUL)   fwd_idx[i] = SymMul(fwd_prog, fwd_idx[(idx_t)op.a], fwd_idx[(idx_t)op.b]);
		else if (op.op == OpKind::DIV)   fwd_idx[i] = SymDiv(fwd_prog, fwd_idx[(idx_t)op.a], fwd_idx[(idx_t)op.b]);
		else if (op.op == OpKind::POW)   fwd_idx[i] = SymPow(fwd_prog, fwd_idx[(idx_t)op.a], fwd_idx[(idx_t)op.b]);
	}

	// Generate symbolic derivative for each parameter
	bind_data->sym_derivs.resize(n_params);
	bind_data->sym_roots.resize(n_params, -1);
	for (idx_t p = 0; p < n_params; p++) {
		SymProg dp;
		int32_t dr = SymDiff(bind_data->prog, bind_data->root, (int32_t)p, dp);
		// dp is self-contained, no fwd_prog needed
		idx_t offset = fwd_prog.size();
		for (auto &sn : dp) {
			if (sn.left  >= 0) sn.left  += (int32_t)offset;
			if (sn.right >= 0) sn.right += (int32_t)offset;
			if (sn.child >= 0) sn.child += (int32_t)offset;
		}
		dr += (int32_t)offset;
		dp.insert(dp.begin(), fwd_prog.begin(), fwd_prog.end());
		bind_data->sym_derivs[p] = std::move(dp);
		bind_data->sym_roots[p]  = dr;
	}

#ifdef DUCKDB_HAVE_LLVM
	// JIT compile tape to native code (forward + backward pass as native code)
	if (!bind_data->prog.empty()) {
		bind_data->jit_func = (void*)CompileRevJITBridge(
			(const void*)&bind_data->prog,
			(const void*)&bind_data->root,
			(const void*)&bind_data->input_node,
			(uint64_t)n_params);
	}
#endif
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

#ifdef DUCKDB_HAVE_LLVM
	// JIT path: fastest — direct native code
	if (bind.jit_func) {
		auto fn = (RevJitFuncType)bind.jit_func;
		std::vector<double> inputs(N), grads(N);
		for (idx_t r = 0; r < count; r++) {
			for (idx_t j = 0; j < N; j++) inputs[j] = in.Get(j, r);
			fn(inputs.data(), grads.data());
			for (idx_t j = 0; j < N; j++) child_ptrs[j][r] = grads[j];
		}
		return;
	}
#endif
	// --- Symbolic path: if sym_derivs were generated at bind time, use them ---
	if (false && !bind.sym_derivs.empty() && (int32_t)bind.sym_derivs.size() == (int32_t)N) {
		// Pre-allocate scratch buffer — reused across all rows and params
		idx_t max_sz = 0;
		for (idx_t j = 0; j < N; j++) if (!bind.sym_derivs[j].empty()) max_sz = std::max(max_sz, (idx_t)bind.sym_derivs[j].size());
		std::vector<double> scratch(max_sz);
		for (idx_t row = 0; row < count; row++) {
			for (idx_t j = 0; j < N; j++) {
				child_ptrs[j][row] = EvalSymLinear(bind.sym_derivs[j], bind.sym_roots[j], in, row, scratch);
			}
		}
		return;
	}

	// --- Tape-based path (fallback) ---
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
			case OpKind::POW:
				for (idx_t r = 0; r < tile_count; r++) vals[i * TILE + r] = std::pow(vals[(idx_t)op.a * TILE + r], vals[(idx_t)op.b * TILE + r]);
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
			case OpKind::POW:
				for (idx_t r = 0; r < tile_count; r++) {
					const double a  = adj[i * TILE + r];
					const double lv = vals[(idx_t)op.a * TILE + r];
					const double rv = vals[(idx_t)op.b * TILE + r];
					adj[(idx_t)op.a * TILE + r] += a * rv * std::pow(lv, rv - 1.0);
					adj[(idx_t)op.b * TILE + r] += (lv > 0.0) ? a * std::pow(lv, rv) * std::log(lv) : 0.0;
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
