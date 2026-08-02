#include "autodiff_stored_lambda.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include <unordered_map>

namespace duckdb {

static int32_t EmitConst(std::vector<StoredCompiledOp> &prog, double v) {
	StoredCompiledOp op; op.op = StoredOpKind::CONST; op.cval = v;
	prog.push_back(op);
	return (int32_t)prog.size() - 1;
}
static int32_t EmitInput(std::vector<StoredCompiledOp> &prog, int32_t slot) {
	StoredCompiledOp op; op.op = StoredOpKind::INPUT; op.input_slot = slot;
	prog.push_back(op);
	return (int32_t)prog.size() - 1;
}
static int32_t EmitNeg(std::vector<StoredCompiledOp> &prog, int32_t a) {
	StoredCompiledOp op; op.op = StoredOpKind::NEG; op.a = a;
	prog.push_back(op);
	return (int32_t)prog.size() - 1;
}
static int32_t EmitBin(std::vector<StoredCompiledOp> &prog, StoredOpKind k, int32_t a, int32_t b) {
	StoredCompiledOp op; op.op = k; op.a = a; op.b = b;
	prog.push_back(op);
	return (int32_t)prog.size() - 1;
}

static bool ParseLambdaHeader(const string &lambda_text, vector<string> &param_names, string &body_text) {
	auto arrow_pos = lambda_text.find("->");
	if (arrow_pos == string::npos) {
		return false;
	}
	string header = lambda_text.substr(0, arrow_pos);
	body_text = lambda_text.substr(arrow_pos + 2);

	StringUtil::Trim(header);
	if (!header.empty() && header.front() == '(' && header.back() == ')') {
		header = header.substr(1, header.size() - 2);
	}
	for (auto &p : StringUtil::Split(header, ',')) {
		string name = p;
		StringUtil::Trim(name);
		if (!name.empty()) {
			param_names.push_back(name);
		}
	}
	StringUtil::Trim(body_text);
	return !param_names.empty() && !body_text.empty();
}

static int32_t CompileParsedExpr(ParsedExpression &expr,
                                 const std::unordered_map<string, idx_t> &name_to_slot,
                                 std::vector<StoredCompiledOp> &prog,
                                 std::vector<int32_t> &input_cache) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::CONSTANT: {
		auto &c = expr.Cast<ConstantExpression>();
		return EmitConst(prog, c.value.GetValue<double>());
	}
	case ExpressionClass::COLUMN_REF: {
		auto &c = expr.Cast<ColumnRefExpression>();
		string nm = c.GetColumnName();
		auto it = name_to_slot.find(nm);
		if (it == name_to_slot.end()) {
			throw BinderException("stored lambda: unknown parameter '" + nm + "'");
		}
		idx_t slot = it->second;
		if (input_cache[slot] < 0) {
			input_cache[slot] = EmitInput(prog, (int32_t)slot);
		}
		return input_cache[slot];
	}
	case ExpressionClass::CAST: {
		auto &c = expr.Cast<CastExpression>();
		return CompileParsedExpr(*c.child, name_to_slot, prog, input_cache);
	}
	case ExpressionClass::FUNCTION: {
		auto &fn = expr.Cast<FunctionExpression>();
		auto &ch = fn.children;
		const string name = StringUtil::Lower(fn.function_name);

		if (ch.size() == 1 && (name == "-" || name.find("neg") != string::npos)) {
			auto a = CompileParsedExpr(*ch[0], name_to_slot, prog, input_cache);
			return EmitNeg(prog, a);
		}
		if (ch.size() == 1 && name == "exp") {
			auto a = CompileParsedExpr(*ch[0], name_to_slot, prog, input_cache);
			StoredCompiledOp o; o.op = StoredOpKind::EXP; o.a = a; prog.push_back(o); return (int32_t)prog.size() - 1;
		}
		if (ch.size() == 1 && (name == "ln" || name == "log")) {
			auto a = CompileParsedExpr(*ch[0], name_to_slot, prog, input_cache);
			StoredCompiledOp o; o.op = StoredOpKind::LOG; o.a = a; prog.push_back(o); return (int32_t)prog.size() - 1;
		}
		if (ch.size() == 2) {
			auto L = CompileParsedExpr(*ch[0], name_to_slot, prog, input_cache);
			auto R = CompileParsedExpr(*ch[1], name_to_slot, prog, input_cache);
			if (name == "+") return EmitBin(prog, StoredOpKind::ADD, L, R);
			if (name == "-") return EmitBin(prog, StoredOpKind::SUB, L, R);
			if (name == "*") return EmitBin(prog, StoredOpKind::MUL, L, R);
			if (name == "/") return EmitBin(prog, StoredOpKind::DIV, L, R);
			if (name == "^" || name == "pow") return EmitBin(prog, StoredOpKind::POW, L, R);
		}
		throw BinderException("stored lambda: unsupported function '" + name + "'");
	}
	case ExpressionClass::OPERATOR: {
		auto &op = expr.Cast<OperatorExpression>();
		auto &ch = op.children;
		if (ch.size() == 1) {
			auto a = CompileParsedExpr(*ch[0], name_to_slot, prog, input_cache);
			return EmitNeg(prog, a);
		}
		throw BinderException("stored lambda: unsupported operator expression");
	}
	default:
		throw BinderException("stored lambda: unsupported expression class");
	}
}

ParsedLambdaTape CompileStoredLambda(const string &lambda_text) {
	vector<string> param_names;
	string body_text;
	if (!ParseLambdaHeader(lambda_text, param_names, body_text)) {
		throw BinderException("stored lambda: could not parse '(params) -> body' from: " + lambda_text);
	}
	idx_t n_params = param_names.size();

	auto expressions = Parser::ParseExpressionList(body_text);
	if (expressions.size() != 1) {
		throw BinderException("stored lambda: body must be exactly one expression");
	}

	std::unordered_map<string, idx_t> name_to_slot;
	for (idx_t i = 0; i < n_params; i++) {
		name_to_slot[param_names[i]] = i;
	}

	ParsedLambdaTape tape;
	tape.n_params = n_params;
	std::vector<int32_t> input_cache(n_params, -1);
	tape.root = CompileParsedExpr(*expressions[0], name_to_slot, tape.prog, input_cache);
	return tape;
}

} // namespace duckdb
