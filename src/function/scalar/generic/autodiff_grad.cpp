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
#include "autodiff_stored_lambda.hpp"

#include <cctype>
#include <cmath>
#include <memory>
#include <vector>
#include <unordered_map>

#define DUCKDB_HAVE_LLVM
// Bridge to JIT implementation in autodiff_jit.cpp (outside duckdb namespace)
#ifdef DUCKDB_HAVE_LLVM
using JitFuncType = void(*)(const double*, double*);
extern JitFuncType CompileJITBridge(const void* prog_ptr, int32_t root, uint64_t N);
#endif

namespace duckdb {


static constexpr bool MYGRAD_DEBUG = false;
static constexpr idx_t MYGRAD_MAX_PARAMS = 32;

// Operator kinds resolved at bind time, used by EvalExpr at execution time.
enum class OpKind { ADD, SUB, MUL, DIV, NEG, UNKNOWN };

// ============================================================================
// Bind data: store the lambda expression + how many lambda parameters it has
// ============================================================================
enum class FwdOpKind : uint8_t { INPUT, CONST, ADD, SUB, MUL, DIV, NEG, POW };

struct FwdOp {
	FwdOpKind op;
	int32_t a = -1;
	int32_t b = -1;
	int32_t input_slot = -1;
	double cval = 0.0;
};


struct AutoDiffGradBindData : public FunctionData {
	unique_ptr<Expression> lambda_expr;
	idx_t param_cnt;
	std::unordered_map<Expression*, OpKind> op_cache;
	std::vector<FwdOp> prog;
	int32_t root = -1;
#ifdef DUCKDB_HAVE_LLVM
	void* jit_func = nullptr;
#endif
	// --- Dynamic (per-row) stored-lambda support ---
	bool is_dynamic_lambda = false;
	struct DynamicTapeEntry {
		std::vector<FwdOp> prog;
		int32_t root = -1;
	};
	mutable std::unordered_map<std::string, std::shared_ptr<DynamicTapeEntry>> dynamic_tape_cache;

	AutoDiffGradBindData(unique_ptr<Expression> expr_p, idx_t nparams_p)
	    : lambda_expr(std::move(expr_p)), param_cnt(nparams_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto out = make_uniq<AutoDiffGradBindData>(lambda_expr ? lambda_expr->Copy() : nullptr, param_cnt);
		out->op_cache = op_cache;
		out->prog = prog;
		out->root = root;
		out->is_dynamic_lambda = is_dynamic_lambda;
		out->dynamic_tape_cache = dynamic_tape_cache;
		return out;
	}

	bool Equals(const FunctionData &) const override {
		return true;
	}
};

// ============================================================================
// Compiled forward-mode tape
// ============================================================================
// ============================================================================
// Dual number for forward mode AD: value + gradient vector
// ============================================================================
struct Dual {
	double val;
	idx_t n;
	double grad[MYGRAD_MAX_PARAMS];
};

static Dual MakeConst(double v, idx_t n) {
	Dual d;
	d.val = v;
	d.n = n;
	for (idx_t i = 0; i < n; i++) {
		d.grad[i] = 0.0;
	}
	return d;
}

static Dual MakeInput(double v, idx_t slot, idx_t n) {
	Dual d;
	d.val = v;
	d.n = n;
	for (idx_t i = 0; i < n; i++) d.grad[i] = 0.0;
	if (slot < n) d.grad[slot] = 1.0;
	return d;
}
static Dual Pow(const Dual &a, const Dual &b, idx_t n) {
	Dual o;
	double pv = std::pow(a.val, b.val);
	o.val = pv;
	o.n = n;
	for (idx_t i = 0; i < n; i++) {
		o.grad[i] = b.val * std::pow(a.val, b.val - 1.0) * a.grad[i];
		if (a.val > 0.0) o.grad[i] += pv * std::log(a.val) * b.grad[i];
	}
	return o;
}
static Dual Add(const Dual &a, const Dual &b) {
	Dual o;
	o.val = a.val + b.val;
	o.n = a.n;
	for (idx_t i = 0; i < a.n; i++) {
		o.grad[i] = a.grad[i] + b.grad[i];
	}
	return o;
}

static Dual Sub(const Dual &a, const Dual &b) {
	Dual o;
	o.val = a.val - b.val;
	o.n = a.n;
	for (idx_t i = 0; i < a.n; i++) {
		o.grad[i] = a.grad[i] - b.grad[i];
	}
	return o;
}

static Dual Mul(const Dual &a, const Dual &b) {
	Dual o;
	o.val = a.val * b.val;
	o.n = a.n;
	for (idx_t i = 0; i < a.n; i++) {
		o.grad[i] = a.grad[i] * b.val + b.grad[i] * a.val;
	}
	return o;
}

static Dual Div(const Dual &a, const Dual &b) {
	Dual o;
	o.val = a.val / b.val;
	o.n = a.n;

	const double inv = 1.0 / b.val;
	const double inv2 = inv * inv;

	for (idx_t i = 0; i < a.n; i++) {
		o.grad[i] = (a.grad[i] * b.val - b.grad[i] * a.val) * inv2;
	}
	return o;
}

static Dual Neg(const Dual &a) {
	Dual o;
	o.val = -a.val;
	o.n = a.n;
	for (idx_t i = 0; i < a.n; i++) {
		o.grad[i] = -a.grad[i];
	}
	return o;
}

// ============================================================================
// Row input: read numeric args for the current row
// ============================================================================
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

	double Get(idx_t col, idx_t row) const {
		return ptrs[col][row];
	}
};

static bool NameToSlot(const string &raw_in, idx_t param_cnt, idx_t &slot_out) {
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
       if (s == "u" || s == "x" || s == "a") { slot_out = 0; return true; }
       if (s == "v" || s == "y" || s == "b") { slot_out = 1; return true; }
       if (s == "w" || s == "z" || s == "c") { slot_out = 2; return true; }
    }


	idx_t pos = 0;
	while (pos < s.size() && !std::isdigit((unsigned char)s[pos])) {
		pos++;
	}
	if (pos == s.size()) {
		return false;
	}

	uint64_t num = 0;
	for (idx_t i = pos; i < s.size(); i++) {
		if (!std::isdigit((unsigned char)s[i])) {
			break;
		}
		num = num * 10 + (uint64_t)(s[i] - '0');
	}
	if (num == 0) {
		return false;
	}

	slot_out = (idx_t)(num - 1);
	return slot_out < param_cnt;
}

// ============================================================================
// Resolve operator kind from an expression (called at bind time only)
// ============================================================================
static OpKind ResolveOp(Expression &expr) {
	if (expr.expression_class == ExpressionClass::BOUND_OPERATOR) {
		auto &op = expr.Cast<BoundOperatorExpression>();
		string t = StringUtil::Lower(ExpressionTypeToString(op.type));
		idx_t nch = op.children.size();
		if (nch == 1 && (t.find("neg") != string::npos || t.find("unary") != string::npos || t == "-")) return OpKind::NEG;
		if (nch == 2) {
			if (t.find("add") != string::npos || t.find("plus") != string::npos) return OpKind::ADD;
			if (t.find("sub") != string::npos || t.find("minus") != string::npos) return OpKind::SUB;
			if (t.find("mul") != string::npos) return OpKind::MUL;
			if (t.find("div") != string::npos) return OpKind::DIV;
		}
	}
	if (expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		string name = StringUtil::Lower(fn.function.name);
		idx_t nch = fn.children.size();
		if (nch == 1 && (name.find("neg") != string::npos || name == "-")) return OpKind::NEG;
		if (nch == 2) {
			if (name == "+" || name.find("add") != string::npos || name.find("plus") != string::npos) return OpKind::ADD;
			if (name == "-" || name.find("sub") != string::npos || name.find("minus") != string::npos) return OpKind::SUB;
			if (name == "*" || name.find("mul") != string::npos) return OpKind::MUL;
			if (name == "/" || name.find("div") != string::npos) return OpKind::DIV;
		}
	}
	return OpKind::UNKNOWN;
}

// ============================================================================
// Walk expression tree once at bind time and fill the operator cache
// ============================================================================
//------------------------------------------------------------------------------
// Forward-mode tape compiler: Expression -> FwdOp prog
//------------------------------------------------------------------------------
static int32_t CompileFwd(Expression &expr, idx_t param_cnt,
                          std::vector<FwdOp> &prog,
                          const vector<unique_ptr<Expression>> &captures) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_CONSTANT: {
		auto &c = expr.Cast<BoundConstantExpression>();
		FwdOp o; o.op = FwdOpKind::CONST; o.cval = c.value.GetValue<double>();
		prog.push_back(o); return (int32_t)prog.size()-1;
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &c = expr.Cast<BoundColumnRefExpression>();
		idx_t slot = c.binding.column_index;
		if (slot >= param_cnt) { FwdOp o; o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1; }
		FwdOp o; o.op=FwdOpKind::INPUT; o.input_slot=(int32_t)slot; prog.push_back(o); return (int32_t)prog.size()-1;
	}
	case ExpressionClass::BOUND_REF: {
		string nm = expr.alias; if (nm.empty()) nm = expr.ToString();
		// Handle capture refs like #N
		if (!nm.empty() && nm[0] == '#') {
			for (idx_t ci = 0; ci < captures.size(); ci++) {
				if (captures[ci]->expression_class == ExpressionClass::BOUND_CONSTANT) {
					auto &cc = captures[ci]->Cast<BoundConstantExpression>();
					FwdOp o; o.op=FwdOpKind::CONST; o.cval=cc.value.GetValue<double>(); prog.push_back(o); return (int32_t)prog.size()-1;
				}
			}
			FwdOp o; o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1;
		}
		// Use NameToSlot to handle x,y,p1,p2,x1,x2 etc.
		idx_t slot;
		if (NameToSlot(nm, param_cnt, slot)) {
			FwdOp o; o.op=FwdOpKind::INPUT; o.input_slot=(int32_t)slot; prog.push_back(o); return (int32_t)prog.size()-1;
		}
		FwdOp o; o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1;
	}
	case ExpressionClass::BOUND_CAST: {
		auto &c = expr.Cast<BoundCastExpression>();
		return CompileFwd(*c.child, param_cnt, prog, captures);
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		auto &ch = op.children;
		const string t = StringUtil::Lower(ExpressionTypeToString(op.type));
		if (ch.size()==1) { auto a=CompileFwd(*ch[0],param_cnt,prog,captures); FwdOp o; o.op=FwdOpKind::NEG; o.a=a; prog.push_back(o); return (int32_t)prog.size()-1; }
		auto L=CompileFwd(*ch[0],param_cnt,prog,captures);
		auto R=CompileFwd(*ch[1],param_cnt,prog,captures);
		FwdOp o;
		if (t.find("add")!=string::npos||t.find("plus")!=string::npos) o.op=FwdOpKind::ADD;
		else if (t.find("sub")!=string::npos||t.find("minus")!=string::npos) o.op=FwdOpKind::SUB;
		else if (t.find("mul")!=string::npos) o.op=FwdOpKind::MUL;
		else if (t.find("div")!=string::npos) o.op=FwdOpKind::DIV;
		else if (t.find("pow")!=string::npos||t=="^") o.op=FwdOpKind::POW;
		else { o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1; }
		o.a=L; o.b=R; prog.push_back(o); return (int32_t)prog.size()-1;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		auto &ch = fn.children;
		const string name = StringUtil::Lower(fn.function.name);
		if (ch.size()==1) { auto a=CompileFwd(*ch[0],param_cnt,prog,captures); FwdOp o; o.op=FwdOpKind::NEG; o.a=a; prog.push_back(o); return (int32_t)prog.size()-1; }
		if (ch.size()==2) {
			auto L=CompileFwd(*ch[0],param_cnt,prog,captures);
			auto R=CompileFwd(*ch[1],param_cnt,prog,captures);
			FwdOp o;
			if (name=="+"||name.find("add")!=string::npos) o.op=FwdOpKind::ADD;
			else if (name=="-"||name.find("sub")!=string::npos) o.op=FwdOpKind::SUB;
			else if (name=="*"||name.find("mul")!=string::npos) o.op=FwdOpKind::MUL;
			else if (name=="/"||name.find("div")!=string::npos) o.op=FwdOpKind::DIV;
			else if (name=="^"||name=="pow"||name.find("pow")!=string::npos) o.op=FwdOpKind::POW;
			else { o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1; }
			o.a=L; o.b=R; prog.push_back(o); return (int32_t)prog.size()-1;
		}
		FwdOp o; o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1;
	}
	default: {
		FwdOp o; o.op=FwdOpKind::CONST; o.cval=0.0; prog.push_back(o); return (int32_t)prog.size()-1;
	}
	}
}

static void PrecomputeOps(Expression &expr, std::unordered_map<Expression*, OpKind> &cache) {
	if (expr.expression_class == ExpressionClass::BOUND_OPERATOR ||
	    expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
		cache[&expr] = ResolveOp(expr);
	}

	if (expr.expression_class == ExpressionClass::BOUND_OPERATOR) {
		auto &op = expr.Cast<BoundOperatorExpression>();
		for (auto &c : op.children) PrecomputeOps(*c, cache);
	} else if (expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		for (auto &c : fn.children) PrecomputeOps(*c, cache);
	} else if (expr.expression_class == ExpressionClass::BOUND_CAST) {
		auto &c = expr.Cast<BoundCastExpression>();
		PrecomputeOps(*c.child, cache);
	}
}

// ============================================================================
// Evaluate expression with forward mode AD
// Supported: constants, refs, casts, + - * /, unary minus
// ============================================================================
static Dual EvalExpr(Expression &expr, const RowInput &in, idx_t row, const AutoDiffGradBindData &bind) {
	const idx_t N = bind.param_cnt;

	switch (expr.expression_class) {
	case ExpressionClass::BOUND_CONSTANT: {
		auto &c = expr.Cast<BoundConstantExpression>();
		return MakeConst(c.value.GetValue<double>(), N);
	}

	case ExpressionClass::BOUND_COLUMN_REF: {

		auto &c = expr.Cast<BoundColumnRefExpression>();
		idx_t slot = c.binding.column_index;

		if (slot >= N) return MakeConst(0.0, N);

		Dual out = MakeConst(in.Get(slot, row), N);
		out.grad[slot] = 1.0;
		return out;
	}

	case ExpressionClass::BOUND_REF: {

		string nm = expr.alias;
		if (nm.empty()) nm = expr.ToString();

		idx_t slot;
		if (!NameToSlot(nm, N, slot)) {

			return MakeConst(0.0, N);
		}

		Dual out = MakeConst(in.Get(slot, row), N);
		out.grad[slot] = 1.0;
		return out;
	}

	case ExpressionClass::BOUND_CAST: {
		auto &c = expr.Cast<BoundCastExpression>();
		return EvalExpr(*c.child, in, row, bind);
	}

	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		auto &ch = op.children;

		auto it = bind.op_cache.find(&expr);
		OpKind kind = (it != bind.op_cache.end()) ? it->second : OpKind::UNKNOWN;

		if (kind == OpKind::NEG) {
			return Neg(EvalExpr(*ch[0], in, row, bind));
		}

		if (ch.size() != 2) {
			throw BinderException("mygrad: only unary minus and binary +,-,*,/ supported");
		}

		auto L = EvalExpr(*ch[0], in, row, bind);
		auto R = EvalExpr(*ch[1], in, row, bind);

		switch (kind) {
		case OpKind::ADD: return Add(L, R);
		case OpKind::SUB: return Sub(L, R);
		case OpKind::MUL: return Mul(L, R);
		case OpKind::DIV: return Div(L, R);
		default: throw BinderException("mygrad: unsupported operator (use +,-,*,/)");
		}
	}

	case ExpressionClass::BOUND_FUNCTION: {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		auto &ch = fn.children;

		auto it = bind.op_cache.find(&expr);
		OpKind kind = (it != bind.op_cache.end()) ? it->second : OpKind::UNKNOWN;

		if (kind == OpKind::NEG && ch.size() == 1) {
			return Neg(EvalExpr(*ch[0], in, row, bind));
		}

		if (ch.size() == 2) {
			auto L = EvalExpr(*ch[0], in, row, bind);
			auto R = EvalExpr(*ch[1], in, row, bind);

			switch (kind) {
			case OpKind::ADD: return Add(L, R);
			case OpKind::SUB: return Sub(L, R);
			case OpKind::MUL: return Mul(L, R);
			case OpKind::DIV: return Div(L, R);
			default: break;
			}
		}

		throw BinderException("mygrad: unsupported function in forward AD");
	}

	default:
		throw BinderException("mygrad: unsupported expression type in forward AD");
	}
}

// ============================================================================
// Dynamic (per-row) stored-lambda support: converter + cache resolver + single-row runner
// ============================================================================
static FwdOpKind ConvertStoredOpKindToFwd(StoredOpKind k) {
	switch (k) {
	case StoredOpKind::INPUT: return FwdOpKind::INPUT;
	case StoredOpKind::CONST: return FwdOpKind::CONST;
	case StoredOpKind::NEG:   return FwdOpKind::NEG;
	case StoredOpKind::ADD:   return FwdOpKind::ADD;
	case StoredOpKind::SUB:   return FwdOpKind::SUB;
	case StoredOpKind::MUL:   return FwdOpKind::MUL;
	case StoredOpKind::DIV:   return FwdOpKind::DIV;
	case StoredOpKind::POW:   return FwdOpKind::POW;
	}
	throw InternalException("stored lambda (fwd): unknown op kind");
}

static std::shared_ptr<AutoDiffGradBindData::DynamicTapeEntry>
ResolveDynamicLambdaFwd(AutoDiffGradBindData &bind, const string &lambda_text, idx_t n_numeric_args) {
	auto it = bind.dynamic_tape_cache.find(lambda_text);
	if (it != bind.dynamic_tape_cache.end()) {
		return it->second;
	}
	auto parsed = CompileStoredLambda(lambda_text);
	if (parsed.n_params != n_numeric_args) {
		throw InvalidInputException("mygrad_fwd: stored lambda has " + std::to_string(parsed.n_params) +
		                             " parameters but " + std::to_string(n_numeric_args) + " numeric args given");
	}
	auto entry = std::make_shared<AutoDiffGradBindData::DynamicTapeEntry>();
	entry->prog.reserve(parsed.prog.size());
	for (auto &sop : parsed.prog) {
		FwdOp op;
		op.op = ConvertStoredOpKindToFwd(sop.op);
		op.a = sop.a;
		op.b = sop.b;
		op.cval = sop.cval;
		op.input_slot = sop.input_slot;
		entry->prog.push_back(op);
	}
	entry->root = parsed.root;
	bind.dynamic_tape_cache[lambda_text] = entry;
	return entry;
}

// Evaluates one row via dual-number forward propagation over a resolved tape.
static void RunFwdTapeSingleRow(const std::vector<FwdOp> &prog, int32_t root,
                                const RowInput &in, idx_t row, idx_t N,
                                double *out_grads) {
	if (root < 0 || prog.empty()) {
		for (idx_t j = 0; j < N; j++) out_grads[j] = 0.0;
		return;
	}
	const idx_t sz = (idx_t)root + 1;
	std::vector<double> val(sz);
	std::vector<double> grad(sz * N);

	for (idx_t i = 0; i < sz; i++) {
		const auto &op = prog[i];
		switch (op.op) {
		case FwdOpKind::INPUT:
			val[i] = in.Get((idx_t)op.input_slot, row);
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = (j == (idx_t)op.input_slot) ? 1.0 : 0.0;
			break;
		case FwdOpKind::CONST:
			val[i] = op.cval;
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = 0.0;
			break;
		case FwdOpKind::NEG:
			val[i] = -val[(idx_t)op.a];
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = -grad[(idx_t)op.a*N+j];
			break;
		case FwdOpKind::ADD:
			val[i] = val[(idx_t)op.a] + val[(idx_t)op.b];
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = grad[(idx_t)op.a*N+j] + grad[(idx_t)op.b*N+j];
			break;
		case FwdOpKind::SUB:
			val[i] = val[(idx_t)op.a] - val[(idx_t)op.b];
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = grad[(idx_t)op.a*N+j] - grad[(idx_t)op.b*N+j];
			break;
		case FwdOpKind::MUL: {
			double lv = val[(idx_t)op.a], rv = val[(idx_t)op.b];
			val[i] = lv * rv;
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = grad[(idx_t)op.a*N+j]*rv + grad[(idx_t)op.b*N+j]*lv;
			break;
		}
		case FwdOpKind::DIV: {
			double lv = val[(idx_t)op.a], rv = val[(idx_t)op.b];
			val[i] = lv / rv;
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = (grad[(idx_t)op.a*N+j]*rv - grad[(idx_t)op.b*N+j]*lv) / (rv*rv);
			break;
		}
		case FwdOpKind::POW: {
			double lv = val[(idx_t)op.a], rv = val[(idx_t)op.b];
			val[i] = std::pow(lv, rv);
			for (idx_t j = 0; j < N; j++) {
				double dl = grad[(idx_t)op.a*N+j], dr = grad[(idx_t)op.b*N+j];
				double term1 = (lv != 0.0) ? rv * std::pow(lv, rv - 1.0) * dl : 0.0;
				double term2 = (lv > 0.0) ? std::pow(lv, rv) * std::log(lv) * dr : 0.0;
				grad[i*N+j] = term1 + term2;
			}
			break;
		}
		default:
			val[i] = 0.0;
			for (idx_t j = 0; j < N; j++) grad[i*N+j] = 0.0;
			break;
		}
	}
	for (idx_t j = 0; j < N; j++) out_grads[j] = grad[(idx_t)root*N+j];
}

// ============================================================================
// Bind: validate, build STRUCT(d1..dN), keep a COPY of lambda expr
// ============================================================================
static unique_ptr<FunctionData>
AutoDiffGradBind(ClientContext &, ScalarFunction &bound_function, vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < 2) {
		throw BinderException("mygrad: expected at least (x..., lambda)");
	}

	idx_t n_numeric_args = arguments.size() - 1;

	if (n_numeric_args > MYGRAD_MAX_PARAMS) {
		throw BinderException("mygrad: too many parameters (max is 32)");
	}



	child_list_t<LogicalType> kids;
	for (idx_t i = 0; i < n_numeric_args; i++) {
		kids.push_back({string("d") + std::to_string(i + 1), LogicalType::DOUBLE});
	}
	bound_function.return_type = LogicalType::STRUCT(std::move(kids));

	auto *last = arguments.back().get();


	if (last->expression_class != ExpressionClass::BOUND_LAMBDA) {
		Expression *unwrapped = last;
		while (unwrapped->expression_class == ExpressionClass::BOUND_CAST) {
			unwrapped = unwrapped->Cast<BoundCastExpression>().child.get();
		}
		if (last->return_type.id() == LogicalTypeId::STORED_LAMBDA &&
		    unwrapped->expression_class == ExpressionClass::BOUND_CONSTANT) {
			auto &c = unwrapped->Cast<BoundConstantExpression>();
			string lambda_text = StringValue::Get(c.value);
			auto parsed = CompileStoredLambda(lambda_text);
			if (parsed.n_params != n_numeric_args) {
				throw BinderException("mygrad_fwd: stored lambda has " + std::to_string(parsed.n_params) +
				                       " parameters but " + std::to_string(n_numeric_args) + " numeric args given");
			}
			auto dummy = make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0));
			auto bind_data = make_uniq<AutoDiffGradBindData>(std::move(dummy), n_numeric_args);

			bind_data->prog.clear();
			bind_data->prog.reserve(parsed.prog.size());
			for (auto &sop : parsed.prog) {
				FwdOp op;
				op.op = ConvertStoredOpKindToFwd(sop.op);
				op.a = sop.a;
				op.b = sop.b;
				op.cval = sop.cval;
				op.input_slot = sop.input_slot;
				bind_data->prog.push_back(op);
			}
			bind_data->root = parsed.root;
#ifdef DUCKDB_HAVE_LLVM
			bind_data->jit_func = (void*)CompileJITBridge((const void*)&bind_data->prog, bind_data->root, (uint64_t)n_numeric_args);
#endif
			return std::move(bind_data);
		}
		if (last->return_type.id() == LogicalTypeId::STORED_LAMBDA) {
			auto dummy = make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0));
			auto bind_data = make_uniq<AutoDiffGradBindData>(std::move(dummy), n_numeric_args);
			bind_data->is_dynamic_lambda = true;
			return std::move(bind_data);
		}
		auto dummy = make_uniq<BoundConstantExpression>(Value::DOUBLE(0.0));
		return make_uniq<AutoDiffGradBindData>(std::move(dummy), n_numeric_args);
	}

	auto &ble = last->Cast<BoundLambdaExpression>();
	idx_t n_params = ble.parameter_count;

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

	auto bind_data = make_uniq<AutoDiffGradBindData>(std::move(lambda_expr), n_params);
	PrecomputeOps(*bind_data->lambda_expr, bind_data->op_cache);
	// Compile expression to flat tape for linear evaluation
	bind_data->prog.clear();
	bind_data->prog.reserve(64);
	bind_data->root = CompileFwd(*bind_data->lambda_expr, n_params, bind_data->prog, ble.captures);
	// Apply CSE to reduce tape size
	{
		std::vector<FwdOp> new_prog;
		std::unordered_map<std::string,int32_t> seen;
		std::vector<int32_t> remap(bind_data->prog.size(),-1);
		new_prog.reserve(bind_data->prog.size());
		for (idx_t i2=0; i2<bind_data->prog.size(); i2++) {
			FwdOp op2 = bind_data->prog[i2];
			int32_t ra=(op2.a>=0)?remap[(idx_t)op2.a]:-1;
			int32_t rb=(op2.b>=0)?remap[(idx_t)op2.b]:-1;
			op2.a=ra; op2.b=rb;
			char buf[64]; snprintf(buf,sizeof(buf),"%d:%d:%d:%d:%.17g",(int)op2.op,ra,rb,op2.input_slot,op2.cval);
			std::string key(buf);
			auto it=seen.find(key);
			if(it!=seen.end()){remap[i2]=it->second;}
			else{int32_t ni=(int32_t)new_prog.size();new_prog.push_back(op2);seen[key]=ni;remap[i2]=ni;}
		}
		bind_data->root=remap[(idx_t)bind_data->root];
		bind_data->prog=std::move(new_prog);
	}
#ifdef DUCKDB_HAVE_LLVM
	bind_data->jit_func = (void*)CompileJITBridge((const void*)&bind_data->prog, bind_data->root, (uint64_t)n_params);
	Printer::Print(bind_data->jit_func ? "[JIT] compiled OK" : "[JIT] failed");
#endif
	return std::move(bind_data);
}

// ============================================================================
// Execute: evaluate per row and output d1..dN into STRUCT
// ============================================================================
static void AutoDiffGradExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = func_expr.bind_info->Cast<AutoDiffGradBindData>();

	idx_t N = bind.param_cnt;
	idx_t count = args.size();
	if (count == 0) return;

	RowInput in(args, N);

	if (bind.is_dynamic_lambda) {
		auto &lambda_vec = args.data.back();
		lambda_vec.Flatten(count);
		auto lambda_ptrs = FlatVector::GetData<string_t>(lambda_vec);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto &dyn_children = StructVector::GetEntries(result);
		for (idx_t j = 0; j < N; j++) dyn_children[j]->SetVectorType(VectorType::FLAT_VECTOR);
		vector<double *> dyn_ptr(N);
		for (idx_t j = 0; j < N; j++) dyn_ptr[j] = FlatVector::GetData<double>(*dyn_children[j]);

		std::vector<double> row_grads(N);
		for (idx_t r = 0; r < count; r++) {
			string lambda_text = lambda_ptrs[r].GetString();
			auto entry = ResolveDynamicLambdaFwd(bind, lambda_text, N);
			RunFwdTapeSingleRow(entry->prog, entry->root, in, r, N, row_grads.data());
			for (idx_t j = 0; j < N; j++) dyn_ptr[j][r] = row_grads[j];
		}
		return;
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);

	for (idx_t j = 0; j < N; j++) {
		children[j]->SetVectorType(VectorType::FLAT_VECTOR);
	}

	vector<double *> out_ptr(N);
	for (idx_t j = 0; j < N; j++) {
		out_ptr[j] = FlatVector::GetData<double>(*children[j]);
	}

	// JIT path: native code execution
#ifdef DUCKDB_HAVE_LLVM
	if (bind.jit_func) {
		auto fn = (JitFuncType)bind.jit_func;
		std::vector<double> inputs(N), grads(N);
		for (idx_t r = 0; r < count; r++) {
			for (idx_t j = 0; j < N; j++) inputs[j] = in.Get(j, r);
			fn(inputs.data(), grads.data());
			for (idx_t j = 0; j < N; j++) out_ptr[j][r] = grads[j];
		}
	} else
#endif
	// Use compiled tape with tiled vectorisation
	if (!bind.prog.empty() && bind.root >= 0) {
		const idx_t sz = (idx_t)bind.root + 1;
		const idx_t TILE = 64; // smaller tile: sz*TILE*N doubles must fit in L2
		// val[i*TILE+r] = value of node i for row r in tile
		// grad[i*TILE*N + r*N + j] = gradient of node i for row r, param j
		std::vector<double> val(sz * TILE);
		std::vector<double> grad(sz * TILE * N);

		for (idx_t tile_start = 0; tile_start < count; tile_start += TILE) {
			const idx_t tc = std::min(TILE, count - tile_start);

			for (idx_t i = 0; i < sz; i++) {
				const auto &op = bind.prog[i];
				switch (op.op) {
				case FwdOpKind::INPUT:
					for (idx_t r = 0; r < tc; r++) {
						val[i*TILE+r] = in.Get((idx_t)op.input_slot, tile_start+r);
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = (j==(idx_t)op.input_slot) ? 1.0 : 0.0;
					}
					break;
				case FwdOpKind::CONST:
					for (idx_t r = 0; r < tc; r++) {
						val[i*TILE+r] = op.cval;
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = 0.0;
					}
					break;
				case FwdOpKind::NEG:
					for (idx_t r = 0; r < tc; r++) {
						val[i*TILE+r] = -val[(idx_t)op.a*TILE+r];
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = -grad[(idx_t)op.a*TILE*N+r*N+j];
					}
					break;
				case FwdOpKind::ADD:
					for (idx_t r = 0; r < tc; r++) {
						val[i*TILE+r] = val[(idx_t)op.a*TILE+r] + val[(idx_t)op.b*TILE+r];
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = grad[(idx_t)op.a*TILE*N+r*N+j] + grad[(idx_t)op.b*TILE*N+r*N+j];
					}
					break;
				case FwdOpKind::SUB:
					for (idx_t r = 0; r < tc; r++) {
						val[i*TILE+r] = val[(idx_t)op.a*TILE+r] - val[(idx_t)op.b*TILE+r];
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = grad[(idx_t)op.a*TILE*N+r*N+j] - grad[(idx_t)op.b*TILE*N+r*N+j];
					}
					break;
				case FwdOpKind::MUL:
					for (idx_t r = 0; r < tc; r++) {
						double lv = val[(idx_t)op.a*TILE+r], rv = val[(idx_t)op.b*TILE+r];
						val[i*TILE+r] = lv * rv;
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = grad[(idx_t)op.a*TILE*N+r*N+j]*rv + grad[(idx_t)op.b*TILE*N+r*N+j]*lv;
					}
					break;
				case FwdOpKind::DIV:
					for (idx_t r = 0; r < tc; r++) {
						double lv = val[(idx_t)op.a*TILE+r], rv = val[(idx_t)op.b*TILE+r];
						double inv = 1.0/rv;
						val[i*TILE+r] = lv * inv;
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = (grad[(idx_t)op.a*TILE*N+r*N+j]*rv - grad[(idx_t)op.b*TILE*N+r*N+j]*lv)*inv*inv;
					}
					break;
				case FwdOpKind::POW:
					for (idx_t r = 0; r < tc; r++) {
						double lv = val[(idx_t)op.a*TILE+r], rv = val[(idx_t)op.b*TILE+r];
						double pv = std::pow(lv, rv);
						val[i*TILE+r] = pv;
						for (idx_t j = 0; j < N; j++) {
							double dv = rv * std::pow(lv, rv-1.0) * grad[(idx_t)op.a*TILE*N+r*N+j];
							if (lv > 0.0) dv += pv * std::log(lv) * grad[(idx_t)op.b*TILE*N+r*N+j];
							grad[i*TILE*N+r*N+j] = dv;
						}
					}
					break;
				default:
					for (idx_t r = 0; r < tc; r++) {
						val[i*TILE+r] = 0.0;
						for (idx_t j = 0; j < N; j++) grad[i*TILE*N+r*N+j] = 0.0;
					}
					break;
				}
			}
			// Emit gradients for this tile
			for (idx_t j = 0; j < N; j++)
				for (idx_t r = 0; r < tc; r++)
					out_ptr[j][tile_start+r] = grad[(idx_t)bind.root*TILE*N+r*N+j];
		}
	} else {
		for (idx_t r = 0; r < count; r++) {
			auto d = EvalExpr(*bind.lambda_expr, in, r, bind);
			for (idx_t j = 0; j < N; j++) out_ptr[j][r] = d.grad[j];
		}
	}
}

// ============================================================================
// Register: expose as mygrad (or mygrad_fwd if you prefer)
// ============================================================================
void RegisterAutoDiffGrad(BuiltinFunctions &set) {
	if (MYGRAD_DEBUG) {
		Printer::Print("[mygrad] registering forward-mode AD function");
	}

	ScalarFunctionSet fset("mygrad_fwd");
	for (idx_t n = 1; n <= 32; n++) {
		vector<LogicalType> args;
		for (idx_t i = 0; i < n; i++) args.push_back(LogicalType::ANY);
		args.push_back(LogicalType::LAMBDA);

		ScalarFunction fn(std::move(args), LogicalType::ANY, AutoDiffGradExecute, AutoDiffGradBind);
		fn.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		fset.AddFunction(std::move(fn));
	}
	for (idx_t n = 1; n <= 32; n++) {
		vector<LogicalType> args;
		for (idx_t i = 0; i < n; i++) args.push_back(LogicalType::ANY);
		args.push_back(LogicalType(LogicalTypeId::STORED_LAMBDA));
		ScalarFunction fn(std::move(args), LogicalType::ANY, AutoDiffGradExecute, AutoDiffGradBind);
		fn.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
		fset.AddFunction(std::move(fn));
	}
	set.AddFunction(fset);
}

} // namespace duckdb
