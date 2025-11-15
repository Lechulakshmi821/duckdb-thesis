// src/function/scalar/generic/mygrad_lambda.cpp
// Forward-mode AD for a simple SQL-lambda body: (x) -> expression

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector.hpp"

#if __has_include("duckdb/function/scalar_function.hpp")
  #include "duckdb/function/scalar_function.hpp"
#else
  #include "duckdb/function/scalar/ScalarFunction.hpp"
#endif

#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/common/helper.hpp" // duckdb::unique_ptr / make_uniq

#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace duckdb;

// ===================== Dual numbers (forward-mode AD) =====================
struct Dual {
	double val = 0.0;
	std::vector<double> d;
	Dual() = default;
	Dual(double v, idx_t n) : val(v), d(n, 0.0) {}
	Dual(double v, idx_t n, idx_t basis) : val(v), d(n, 0.0) { if (basis < n) d[basis] = 1.0; }

	static Dual add(const Dual &a, const Dual &b) {
		Dual r; r.val = a.val + b.val; r.d.resize(a.d.size(), 0.0);
		for (idx_t i = 0; i < a.d.size(); i++) r.d[i] = a.d[i] + b.d[i];
		return r;
	}
	static Dual sub(const Dual &a, const Dual &b) {
		Dual r; r.val = a.val - b.val; r.d.resize(a.d.size(), 0.0);
		for (idx_t i = 0; i < a.d.size(); i++) r.d[i] = a.d[i] - b.d[i];
		return r;
	}
	static Dual mul(const Dual &a, const Dual &b) {
		Dual r; r.val = a.val * b.val; r.d.resize(a.d.size(), 0.0);
		for (idx_t i = 0; i < a.d.size(); i++) r.d[i] = a.d[i] * b.val + b.d[i] * a.val;
		return r;
	}
	static Dual div(const Dual &a, const Dual &b) {
		Dual r; r.val = a.val / b.val; r.d.resize(a.d.size(), 0.0);
		const double inv = 1.0 / (b.val * b.val);
		for (idx_t i = 0; i < a.d.size(); i++) r.d[i] = (a.d[i] * b.val - b.d[i] * a.val) * inv;
		return r;
	}
	static Dual pow(const Dual &a, const Dual &b) {
		Dual r; r.val = std::pow(a.val, b.val); r.d.resize(a.d.size(), 0.0);
		const double ln_a = std::log(a.val);
		for (idx_t i = 0; i < a.d.size(); i++)
			r.d[i] = r.val * (b.d[i] * ln_a + (a.d[i] * b.val) / a.val);
		return r;
	}
};

// ===================== Tokenizer / Parser =====================
enum class TokKind { End, Num, Id, LParen, RParen, Comma, Plus, Minus, Star, Slash };

struct Token { TokKind kind = TokKind::End; std::string text; double num = 0.0; };

struct Lexer {
	const std::string &s; size_t i = 0; Token t;
	explicit Lexer(const std::string &src) : s(src) { Next(); }

	void Skip() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++; }

	void Next() {
		Skip();
		if (i >= s.size()) { t = {TokKind::End, ""}; return; }
		const char c = s[i];
		if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
			size_t j = i;
			while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.')) j++;
			t.text = s.substr(i, j - i); t.kind = TokKind::Num; t.num = std::stod(t.text); i = j; return;
		}
		if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
			size_t j = i;
			while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_' || s[j] == '.')) j++;
			t.text = s.substr(i, j - i); t.kind = TokKind::Id; i = j; return;
		}
		i++;
		switch (c) {
			case '(': t = {TokKind::LParen, "("}; break;
			case ')': t = {TokKind::RParen, ")"}; break;
			case ',': t = {TokKind::Comma, ","}; break;
			case '+': t = {TokKind::Plus, "+"}; break;
			case '-': t = {TokKind::Minus, "-"}; break;
			case '*': t = {TokKind::Star, "*"}; break;
			case '/': t = {TokKind::Slash, "/"}; break;
			default: throw std::runtime_error("Unexpected character in lambda: " + std::string(1, c));
		}
	}
};

struct AST {
	virtual ~AST() = default;
	virtual Dual Eval(const std::unordered_map<std::string, Dual> &env, idx_t grad_size) const = 0;
};

struct NumAST : AST {
	double v; explicit NumAST(double v) : v(v) {}
	Dual Eval(const std::unordered_map<std::string, Dual>&, idx_t grad_size) const override { return Dual{v, grad_size}; }
};

struct IdAST : AST {
	std::string name; explicit IdAST(std::string n) : name(std::move(n)) {}
	Dual Eval(const std::unordered_map<std::string, Dual> &env, idx_t) const override {
		auto it = env.find(name);
		if (it == env.end()) throw std::runtime_error("Unknown identifier: " + name);
		return it->second;
	}
};

struct BinAST : AST {
	char op; std::unique_ptr<AST> a, b;
	BinAST(char op, std::unique_ptr<AST> a, std::unique_ptr<AST> b) : op(op), a(std::move(a)), b(std::move(b)) {}
	Dual Eval(const std::unordered_map<std::string, Dual> &env, idx_t n) const override {
		auto A = a->Eval(env, n); auto B = b->Eval(env, n);
		switch (op) {
			case '+': return Dual::add(A,B);
			case '-': return Dual::sub(A,B);
			case '*': return Dual::mul(A,B);
			case '/': return Dual::div(A,B);
			default: throw std::runtime_error("Unknown binary op");
		}
	}
};

struct NegAST : AST {
	std::unique_ptr<AST> x; explicit NegAST(std::unique_ptr<AST> x) : x(std::move(x)) {}
	Dual Eval(const std::unordered_map<std::string, Dual> &env, idx_t n) const override {
		auto v = x->Eval(env, n); v.val = -v.val; for (auto &g : v.d) g = -g; return v;
	}
};

struct PowAST : AST {
	std::unique_ptr<AST> a, b; PowAST(std::unique_ptr<AST> a, std::unique_ptr<AST> b) : a(std::move(a)), b(std::move(b)) {}
	Dual Eval(const std::unordered_map<std::string, Dual> &env, idx_t n) const override {
		auto A = a->Eval(env, n); auto B = b->Eval(env, n); return Dual::pow(A,B);
	}
};

struct Parser {
	Lexer lex; explicit Parser(const std::string &s) : lex(s) {}
	std::unique_ptr<AST> Parse() {
		const std::string &src = lex.s; auto pos = src.find("->");
		if (pos != std::string::npos) return Parser(src.substr(pos + 2)).ParseExpr();
		return ParseExpr();
	}
	std::unique_ptr<AST> ParseExpr() {
		auto lhs = ParseTerm();
		while (lex.t.kind == TokKind::Plus || lex.t.kind == TokKind::Minus) {
			char op = (lex.t.kind == TokKind::Plus) ? '+' : '-'; lex.Next(); auto rhs = ParseTerm();
			lhs = std::make_unique<BinAST>(op, std::move(lhs), std::move(rhs));
		}
		return lhs;
	}
	std::unique_ptr<AST> ParseTerm() {
		auto lhs = ParseUnary();
		while (lex.t.kind == TokKind::Star || lex.t.kind == TokKind::Slash) {
			char op = (lex.t.kind == TokKind::Star) ? '*' : '/'; lex.Next(); auto rhs = ParseUnary();
			lhs = std::make_unique<BinAST>(op, std::move(lhs), std::move(rhs));
		}
		return lhs;
	}
	std::unique_ptr<AST> ParseUnary() {
		if (lex.t.kind == TokKind::Minus) { lex.Next(); return std::make_unique<NegAST>(ParseUnary()); }
		return ParsePrimary();
	}
	std::unique_ptr<AST> ParsePrimary() {
		if (lex.t.kind == TokKind::Num) { double v = lex.t.num; lex.Next(); return std::make_unique<NumAST>(v); }
		if (lex.t.kind == TokKind::Id) {
			std::string name = lex.t.text; lex.Next();
			if (name == "pow" && lex.t.kind == TokKind::LParen) {
				lex.Next(); auto a = ParseExpr();
				if (lex.t.kind != TokKind::Comma) throw std::runtime_error("Expected ',' in pow(a,b)");
				lex.Next(); auto b = ParseExpr();
				if (lex.t.kind != TokKind::RParen) throw std::runtime_error("Expected ')' in pow(a,b)");
				lex.Next(); return std::make_unique<PowAST>(std::move(a), std::move(b));
			}
			auto dot = name.find('.'); if (dot != std::string::npos) name = name.substr(dot + 1); // x.a -> a
			return std::make_unique<IdAST>(name);
		}
		if (lex.t.kind == TokKind::LParen) { lex.Next(); auto e = ParseExpr(); if (lex.t.kind != TokKind::RParen) throw std::runtime_error("Expected ')'"); lex.Next(); return e; }
		throw std::runtime_error("Unexpected token while parsing expression");
	}
};

// ===================== Bind data =====================
struct MyGradBind : public FunctionData {
	std::vector<std::string> field_names;
	duckdb::unique_ptr<FunctionData> Copy() const override {
		auto res = duckdb::make_uniq<MyGradBind>(); res->field_names = field_names; return res;
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = (const MyGradBind&)other_p; return field_names == o.field_names;
	}
};

// Return type: STRUCT( fields..., d<field>..., result DOUBLE )
static LogicalType BuildReturnType(const std::vector<std::string> &fields) {
	child_list_t<LogicalType> children;
	for (auto &f : fields) children.emplace_back(f, LogicalType::DOUBLE);
	for (auto &f : fields) children.emplace_back("d" + f, LogicalType::DOUBLE);
	children.emplace_back("result", LogicalType::DOUBLE);
	return LogicalType::STRUCT(children);
}

static duckdb::unique_ptr<FunctionData>
MyGradBindFunc(ClientContext &, ScalarFunction &fun, duckdb::vector<duckdb::unique_ptr<Expression>> &args) {
	if (args.size() != 2) throw BinderException("mygrad_lambda expects (STRUCT, VARCHAR)");
	if (args[0]->return_type.id() != LogicalTypeId::STRUCT) {
		throw BinderException("mygrad_lambda: first argument must be a STRUCT of DOUBLE fields");
	}

	std::vector<std::string> names;
	for (auto &c : StructType::GetChildTypes(args[0]->return_type)) names.emplace_back(c.first);
	fun.return_type = BuildReturnType(names);

	auto bind = duckdb::make_uniq<MyGradBind>(); bind->field_names = std::move(names); return bind;
}

// ===================== Exec =====================
static void MyGradExec(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bfe  = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = bfe.bind_info->Cast<MyGradBind>();

	const idx_t nrows = args.size();
	const idx_t nvars = bind.field_names.size();

	// Inputs
	Vector &in_struct   = args.data[0];
	auto   &in_children = StructVector::GetEntries(in_struct);
	D_ASSERT(in_children.size() == nvars);

	UnifiedVectorFormat lambda_uvf;
	args.data[1].ToUnifiedFormat(nrows, lambda_uvf);
	auto lambda_data = UnifiedVectorFormat::GetData<string_t>(lambda_uvf);

	// Output
	auto &out_entries = StructVector::GetEntries(result);
	D_ASSERT(out_entries.size() == (2*nvars + 1));
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &out_valid = FlatVector::Validity(result);
	out_valid.SetAllValid(nrows);

	// Prepare per-column access
	std::vector<UnifiedVectorFormat> in_uvf(nvars);
	for (idx_t f = 0; f < nvars; f++) in_children[f]->ToUnifiedFormat(nrows, in_uvf[f]);

	for (idx_t row = 0; row < nrows; row++) {
		const auto ridx = lambda_uvf.sel->get_index(row);
		if (!lambda_uvf.validity.RowIsValid(ridx)) { out_valid.SetInvalid(row); continue; }

		// Lambda body
		const std::string expr_src = lambda_data[ridx].GetString();

		// Parse
		std::unique_ptr<AST> ast;
		try { Parser p(expr_src); ast = p.Parse(); }
		catch (const std::exception &e) { throw InvalidInputException("mygrad_lambda: parse error: %s", e.what()); }

		// Build environment
		std::unordered_map<std::string, Dual> env; env.reserve(nvars);
		std::vector<double> point(nvars, 0.0);
		bool any_null = false;

		for (idx_t f = 0; f < nvars; f++) {
			auto &uvf = in_uvf[f];
			const auto pi = uvf.sel->get_index(row);
			if (!uvf.validity.RowIsValid(pi)) { any_null = true; break; }
			const double *col = UnifiedVectorFormat::GetData<double>(uvf);
			const double v = col[pi];
			point[f] = v;
			env[bind.field_names[f]] = Dual(v, nvars, f);
		}
		if (any_null) { out_valid.SetInvalid(row); continue; }

		// Eval & write
		Dual y;
		try { y = ast->Eval(env, nvars); }
		catch (const std::exception &e) { throw InvalidInputException("mygrad_lambda: eval error: %s", e.what()); }

		for (idx_t f = 0; f < nvars; f++)  FlatVector::GetData<double>(*out_entries[f])[row]         = point[f];
		for (idx_t f = 0; f < nvars; f++)  FlatVector::GetData<double>(*out_entries[nvars + f])[row] = (f < y.d.size() ? y.d[f] : 0.0);
		                                   FlatVector::GetData<double>(*out_entries[2*nvars])[row]   = y.val;
	}
}

// ===================== Factory (export only; register elsewhere) =====================
namespace duckdb {
ScalarFunction MakeMyGradLambda() {
	ScalarFunction fun(
		duckdb::vector<LogicalType>{ LogicalType::ANY, LogicalType::VARCHAR },
		LogicalType::ANY,
		MyGradExec
	);
	fun.bind = MyGradBindFunc;
	fun.side_effects = FunctionSideEffects::NO_SIDE_EFFECTS;
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	return fun;
}
} // namespace duckdb

// ---- local registration shim (provides the symbol the registry calls)
#if __has_include("duckdb/function/builtins.hpp")
  #include "duckdb/function/builtins.hpp"
#endif

namespace duckdb {

// Factory is defined above in this file:
ScalarFunction MakeMyGradLambda();

// This satisfies the linker: called from BuiltinFunctions::RegisterGenericFunctions()
void RegisterMyGradLambda(BuiltinFunctions &set) {
    ScalarFunctionSet fset("mygrad_lambda");
    fset.AddFunction(MakeMyGradLambda());
    set.AddFunction(fset);
}

} // namespace duckdb
