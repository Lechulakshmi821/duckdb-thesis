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
#include <unordered_map>

namespace duckdb {


static constexpr bool MYGRAD_DEBUG = false;
static constexpr idx_t MYGRAD_MAX_PARAMS = 32;

// Operator kinds resolved at bind time, used by EvalExpr at execution time.
enum class OpKind { ADD, SUB, MUL, DIV, NEG, UNKNOWN };

// ============================================================================
// Bind data: store the lambda expression + how many lambda parameters it has
// ============================================================================
struct AutoDiffGradBindData : public FunctionData {
	unique_ptr<Expression> lambda_expr;
	idx_t param_cnt;
	std::unordered_map<Expression*, OpKind> op_cache;

	AutoDiffGradBindData(unique_ptr<Expression> expr_p, idx_t nparams_p)
	    : lambda_expr(std::move(expr_p)), param_cnt(nparams_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AutoDiffGradBindData>(lambda_expr ? lambda_expr->Copy() : nullptr, param_cnt);
	}

	bool Equals(const FunctionData &) const override {
		return true;
	}
};

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

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);

	for (idx_t j = 0; j < N; j++) {
		children[j]->SetVectorType(VectorType::FLAT_VECTOR);
	}

	vector<double *> out_ptr(N);
	for (idx_t j = 0; j < N; j++) {
		out_ptr[j] = FlatVector::GetData<double>(*children[j]);
	}

	for (idx_t r = 0; r < count; r++) {
		auto d = EvalExpr(*bind.lambda_expr, in, r, bind);
		for (idx_t j = 0; j < N; j++) {
			out_ptr[j][r] = d.grad[j];
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
	set.AddFunction(fset);
}

} // namespace duckdb
