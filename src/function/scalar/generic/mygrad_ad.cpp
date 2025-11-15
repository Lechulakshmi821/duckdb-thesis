#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/lambda_expression.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {

// ===========================================================================
// Helpers
// ===========================================================================
static inline double AsDouble(const Value &v) {
	if (v.IsNull()) return 0.0;
	Value x = v;
	if (x.type() != LogicalType::DOUBLE) {
		x = x.DefaultCastAs(LogicalType::DOUBLE);
	}
	return x.GetValue<double>();
}

static inline std::string ToLower(std::string s) {
	for (auto &ch : s) ch = (char)std::tolower((unsigned char)ch);
	return s;
}

static inline std::string StripQuotes(const std::string &s) {
	if (s.size() >= 2 && ((s.front()=='\'' && s.back()=='\'') || (s.front()=='"' && s.back()=='"'))) {
		return s.substr(1, s.size()-2);
	}
	return s;
}

static inline std::string TrimCopy(std::string s) {
	StringUtil::Trim(s); // in-place
	return s;
}

// ===========================================================================
// Optional lambda header parsing for the TABLE FUNCTION string interface
// Syntax: "(x,y,price) -> body"   OR   just "body"
// ===========================================================================
static void ParseLambdaHeaderIfAny(const std::string &expr_in,
                                   std::vector<std::string> &param_names,
                                   std::string &body_out) {
	std::string s = expr_in; // make mutable
	StringUtil::Trim(s);

	const auto arrow = s.find("->");
	if (arrow == std::string::npos) {
		// No header; whole string is the body
		body_out = s;
		return;
	}

	std::string head = s.substr(0, arrow);
	std::string body = s.substr(arrow + 2);
	StringUtil::Trim(head);
	StringUtil::Trim(body);

	if (head.size() < 2 || head.front() != '(' || head.back() != ')') {
		throw BinderException("mygrad_ad_tf: lambda header must be '(...) -> body'");
	}

	std::string inside = head.substr(1, head.size() - 2);
	StringUtil::Trim(inside);

	param_names.clear();
	if (!inside.empty()) {
		auto parts = StringUtil::Split(inside, ',');
		for (auto &p : parts) {
			auto id = TrimCopy(p);
			if (id.empty()) {
				throw BinderException("mygrad_ad_tf: empty parameter name in lambda header");
			}
			param_names.push_back(ToLower(id));
		}
	}

	if (body.empty()) {
		throw BinderException("mygrad_ad_tf: lambda body is empty");
	}
	body_out = body;
}

// ===========================================================================
// Tokenizer for tiny expression grammar
// ===========================================================================
enum class TokType { END, IDENT, NUMBER, PLUS, MINUS, STAR, SLASH, LPAREN, RPAREN, COMMA };
struct Token { TokType type; std::string text; double number; };

struct Lexer {
	const std::string &s; idx_t i = 0;
	explicit Lexer(const std::string &str) : s(str) {}
	static bool IsIdentStart(char c){ return std::isalpha((unsigned char)c) || c=='_'; }
	static bool IsIdentChar (char c){ return std::isalnum((unsigned char)c) || c=='_'; }

	Token Next() {
		while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
		if (i >= s.size()) return {TokType::END,"",0.0};
		char c = s[i];
		if (c=='+'){ i++; return {TokType::PLUS, "+", 0.0}; }
		if (c=='-'){ i++; return {TokType::MINUS,"-", 0.0}; }
		if (c=='*'){ i++; return {TokType::STAR, "*", 0.0}; }
		if (c=='/'){ i++; return {TokType::SLASH,"/", 0.0}; }
		if (c=='('){ i++; return {TokType::LPAREN,"(",0.0}; }
		if (c==')'){ i++; return {TokType::RPAREN,")",0.0}; }
		if (c==','){ i++; return {TokType::COMMA, ",",0.0}; }

		if (std::isdigit((unsigned char)c) || c=='.') {
			idx_t start=i;
			while (i<s.size() && std::isdigit((unsigned char)s[i])) i++;
			if (i<s.size() && s[i]=='.'){ i++; while (i<s.size() && std::isdigit((unsigned char)s[i])) i++; }
			if (i<s.size() && (s[i]=='e'||s[i]=='E')) {
				idx_t j=i+1; if (j<s.size() && (s[j]=='+'||s[j]=='-')) j++;
				bool expd=false; while (j<s.size() && std::isdigit((unsigned char)s[j])){ expd=true; j++; }
				if (expd) i=j;
			}
			auto t = s.substr(start, i-start);
			char *endp=nullptr; double val = std::strtod(t.c_str(), &endp);
			return {TokType::NUMBER, t, val};
		}
		if (IsIdentStart(c)) {
			idx_t start=i++; while (i<s.size() && IsIdentChar(s[i])) i++;
			return {TokType::IDENT, s.substr(start, i-start), 0.0};
		}
		throw BinderException("mygrad_ad_tf: invalid character '%c' in expression", c);
	}
};

// ===========================================================================
// Dual numbers + AST
// ===========================================================================
struct Dual {
	double v;
	std::vector<double> g;
	explicit Dual(idx_t p):v(0.0),g(p,0.0){}
};

struct Node {
	virtual ~Node()=default;
	virtual Dual eval(const std::vector<double>&, idx_t) const=0;
};

struct NConst : Node {
	double c;
	explicit NConst(double c_):c(c_){}
	Dual eval(const std::vector<double>&, idx_t p) const override { Dual r(p); r.v=c; return r; }
};

struct NVar : Node {
	int id; // 0..p-1
	explicit NVar(int i):id(i){}
	Dual eval(const std::vector<double> &x, idx_t p) const override { Dual r(p); r.v=x[id]; r.g[id]=1.0; return r; }
};

struct NUnaryMinus : Node {
	std::unique_ptr<Node> c;
	explicit NUnaryMinus(std::unique_ptr<Node> n):c(std::move(n)){}
	Dual eval(const std::vector<double> &x, idx_t p) const override {
		auto a=c->eval(x,p); a.v=-a.v; for(auto &gi:a.g) gi=-gi; return a;
	}
};

struct NBin : Node {
	std::unique_ptr<Node> L,R; char op; // '+','-','*','/','^'
	NBin(std::unique_ptr<Node> l, char o, std::unique_ptr<Node> r):L(std::move(l)),R(std::move(r)),op(o){}
	Dual eval(const std::vector<double> &x, idx_t p) const override {
		auto a=L->eval(x,p), b=R->eval(x,p); Dual r(p);
		switch(op){
			case '+': r.v=a.v+b.v; for(idx_t i=0;i<p;i++) r.g[i]=a.g[i]+b.g[i]; return r;
			case '-': r.v=a.v-b.v; for(idx_t i=0;i<p;i++) r.g[i]=a.g[i]-b.g[i]; return r;
			case '*': r.v=a.v*b.v; for(idx_t i=0;i<p;i++) r.g[i]=a.g[i]*b.v + a.v*b.g[i]; return r;
			case '/':
				if (b.v==0.0) throw BinderException("mygrad_ad_tf: division by zero");
				r.v=a.v/b.v; for(idx_t i=0;i<p;i++) r.g[i]=(a.g[i]*b.v - a.v*b.g[i])/(b.v*b.v); return r;
			case '^': {
				if (a.v<=0.0) throw BinderException("mygrad_ad_tf: pow base must be > 0 (got %g)", a.v);
				double f=std::pow(a.v,b.v), ln_a=std::log(a.v);
				r.v=f; for(idx_t i=0;i<p;i++) r.g[i]=f*( b.g[i]*ln_a + b.v*(a.g[i]/a.v) ); return r;
			}
			default: throw BinderException("mygrad_ad_tf: internal unknown op");
		}
	}
};

// ===========================================================================
// Recursive-descent parser with symbol table
// ===========================================================================
struct Parser {
	Lexer lex; Token cur;
	const std::unordered_map<std::string,int> &sym; // name -> index

	explicit Parser(const std::string&s, const std::unordered_map<std::string,int> &sym_):lex(s),sym(sym_){ cur=lex.Next(); }
	void eat(TokType t){ if(cur.type!=t) throw BinderException("mygrad_ad_tf: syntax error near '%s'", cur.text.c_str()); cur=lex.Next(); }

	std::unique_ptr<Node> expr(){ auto n=term(); while(cur.type==TokType::PLUS||cur.type==TokType::MINUS){ char o=(cur.type==TokType::PLUS?'+':'-'); eat(cur.type); auto r=term(); n=std::make_unique<NBin>(std::move(n),o,std::move(r)); } return n; }
	std::unique_ptr<Node> term(){ auto n=power(); while(cur.type==TokType::STAR||cur.type==TokType::SLASH){ char o=(cur.type==TokType::STAR?'*':'/'); eat(cur.type); auto r=power(); n=std::make_unique<NBin>(std::move(n),o,std::move(r)); } return n; }
	std::unique_ptr<Node> power(){
		if (cur.type==TokType::IDENT && ToLower(cur.text)=="pow"){ eat(TokType::IDENT); eat(TokType::LPAREN); auto a=expr(); eat(TokType::COMMA); auto b=expr(); eat(TokType::RPAREN); return std::make_unique<NBin>(std::move(a),'^',std::move(b)); }
		return unary();
	}
	std::unique_ptr<Node> unary(){ if(cur.type==TokType::MINUS){ eat(TokType::MINUS); auto c=unary(); return std::make_unique<NUnaryMinus>(std::move(c)); } return primary(); }
	std::unique_ptr<Node> primary(){
		switch(cur.type){
			case TokType::NUMBER: { double v=cur.number; eat(TokType::NUMBER); return std::make_unique<NConst>(v); }
			case TokType::IDENT: {
				auto name=ToLower(cur.text); eat(TokType::IDENT);
				auto it = sym.find(name);
				if (it==sym.end()) {
					throw BinderException("mygrad_ad_tf: unknown identifier '%s'", name.c_str());
				}
				return std::make_unique<NVar>(it->second);
			}
			case TokType::LPAREN: { eat(TokType::LPAREN); auto n=expr(); eat(TokType::RPAREN); return n; }
			default: throw BinderException("mygrad_ad_tf: unexpected token '%s'", cur.text.c_str());
		}
	}
};

// ===========================================================================
// TABLE FUNCTION: mygrad_ad_tf
//   args: 1..N DOUBLEs, last arg VARCHAR (body or "(params)->body")
//   returns: x1..xN, dx1..dxN, result
// ===========================================================================
struct MyBindDataTF : public FunctionData {
	idx_t pcount = 0;
	std::vector<double> x;
	std::string expr_body;
	std::vector<std::string> names; // parameter names (x1.. or from header)
	std::unique_ptr<Node> ast;

	unique_ptr<FunctionData> Copy() const override {
		auto res = make_uniq<MyBindDataTF>();
		res->pcount   = pcount;
		res->x        = x;
		res->expr_body= expr_body;
		res->names    = names;

		std::unordered_map<std::string,int> sym;
		for (idx_t i=0;i<pcount;i++) sym[names[i]] = int(i);
		Parser p(expr_body, sym);
		res->ast = p.expr();
		return res;
	}
	bool Equals(const FunctionData &o) const override {
		auto &b = o.Cast<const MyBindDataTF>();
		return pcount==b.pcount && x==b.x && expr_body==b.expr_body && names==b.names;
	}
};

static unique_ptr<FunctionData> MygradBindTF(ClientContext &, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names_out) {

	const idx_t argc = input.inputs.size();
	if (argc < 2) {
		throw BinderException("mygrad_ad_tf expects at least 2 arguments: 1..N DOUBLEs, then expression string");
	}

	std::string expr_raw = StripQuotes(input.inputs.back().ToString());
	idx_t p = argc - 1;

	std::vector<double> x(p, 0.0);
	for (idx_t i = 0; i < p; i++) x[i] = AsDouble(input.inputs[i]);

	// Parse optional lambda header
	std::vector<std::string> header_names;
	std::string body;
	ParseLambdaHeaderIfAny(expr_raw, header_names, body);

	std::vector<std::string> param_names;
	if (!header_names.empty()) {
		if (header_names.size() != p) {
			throw BinderException("mygrad_ad_tf: lambda header lists %llu parameters but %llu values were provided",
			                      (unsigned long long)header_names.size(), (unsigned long long)p);
		}
		for (auto &nm : header_names) param_names.push_back(ToLower(nm));
	} else {
		param_names.reserve(p);
		for (idx_t i=0;i<p;i++) param_names.push_back("x"+std::to_string(i+1));
	}

	// Build symbol table
	std::unordered_map<std::string,int> sym;
	for (idx_t i=0;i<p;i++) sym[param_names[i]] = int(i);
	// legacy short names a,b,c (optional)
	if (p >= 1) sym["a"] = 0;
	if (p >= 2) sym["b"] = 1;
	if (p >= 3) sym["c"] = 2;

	Parser parser(body, sym);
	auto ast = parser.expr();

	// Output schema: x1..xN, dx1..dxN, result
	return_types.clear(); names_out.clear();
	for (idx_t i=0;i<p;i++) { return_types.push_back(LogicalType::DOUBLE); names_out.emplace_back(param_names[i]); }
	for (idx_t i=0;i<p;i++) { return_types.push_back(LogicalType::DOUBLE); names_out.emplace_back("d"+param_names[i]); }
	return_types.push_back(LogicalType::DOUBLE); names_out.emplace_back("result");

	auto bind = make_uniq<MyBindDataTF>();
	bind->pcount    = p;
	bind->x         = std::move(x);
	bind->expr_body = std::move(body);
	bind->names     = std::move(param_names);
	bind->ast       = std::move(ast);
	return std::move(bind);
}

struct MyGlobalStateTF final : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override { return 1; }
};
static unique_ptr<GlobalTableFunctionState> MygradInitGlobalTF(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<MyGlobalStateTF>();
}
static unique_ptr<LocalTableFunctionState> MygradInitLocalTF(ExecutionContext &, TableFunctionInitInput &, GlobalTableFunctionState *) {
	return nullptr;
}

static void MygradExecTF(ClientContext &, TableFunctionInput &tinput, DataChunk &output) {
	auto &bind = tinput.bind_data->Cast<MyBindDataTF>();
	auto &g    = tinput.global_state->Cast<MyGlobalStateTF>();
	if (g.done) { output.SetCardinality(0); return; }
	g.done = true;

	auto res = bind.ast->eval(bind.x, bind.pcount);

	for (idx_t c = 0; c < output.ColumnCount(); c++) {
		output.data[c].SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::Validity(output.data[c]).SetAllValid(1);
	}

	idx_t col = 0;
	for (idx_t i = 0; i < bind.pcount; i++) {
		auto *ptr = FlatVector::GetData<double>(output.data[col++]); ptr[0] = bind.x[i];
	}
	for (idx_t i = 0; i < bind.pcount; i++) {
		auto *ptr = FlatVector::GetData<double>(output.data[col++]); ptr[0] = res.g[i];
	}
	{
		auto *ptr = FlatVector::GetData<double>(output.data[col++]); ptr[0] = res.v;
	}
	output.SetCardinality(1);
}

void RegisterMyGradADTF(BuiltinFunctions &set) {
	// Single vararg signature; binder checks shapes
	TableFunction tf(
		"mygrad_ad_tf",
		{LogicalType::ANY},
		MygradExecTF,
		MygradBindTF,
		MygradInitGlobalTF,
		MygradInitLocalTF
	);
	tf.varargs = LogicalType::ANY;
	set.AddFunction(tf);
}

// ===========================================================================
// SCALAR FUNCTIONS with real SQL LAMBDA
//   mygrad_eval(LIST<DOUBLE>, LAMBDA) -> DOUBLE
//   mygrad_grad(LIST<DOUBLE>, LAMBDA) -> STRUCT(result DOUBLE, grad LIST<DOUBLE>)
// ===========================================================================

using ParamMap = std::unordered_map<std::string, int>;

struct CompiledLambda {
	std::vector<std::string> params;   // param names in order
	std::string body_sql;              // normalized body
	std::unique_ptr<Node> ast;         // compiled AST
};

static CompiledLambda CompileDuckDBLambda(const LambdaExpression &lam) {
	CompiledLambda out;

	// Parameters: in newer DuckDB, LambdaExpression::parameters is vector<string>
	// If your version stores expressions, adapt: out.params.push_back(StringUtil::Lower(p->ToString()));
	for (auto &pname : lam.parameters) {
		out.params.push_back(StringUtil::Lower(pname));
	}

	// Body SQL as string
	std::string body_sql = lam.expression->ToString();
	body_sql = ToLower(TrimCopy(body_sql));

	// Symbol table
	ParamMap sym;
	for (idx_t i = 0; i < out.params.size(); i++) sym[out.params[i]] = int(i);

	// Parse to AST
	Parser parser(body_sql, sym);
	out.ast = parser.expr();
	out.body_sql = std::move(body_sql);
	return out;
}

// ---------- mygrad_eval ----------
struct MyGradEvalBind : public FunctionData {
	CompiledLambda cl;
	unique_ptr<FunctionData> Copy() const override {
		auto res = make_uniq<MyGradEvalBind>();
		// Rebuild AST from stored body_sql/params
		res->cl.params = cl.params;
		res->cl.body_sql = cl.body_sql;
		ParamMap sym;
		for (idx_t i=0;i<res->cl.params.size();i++) sym[res->cl.params[i]] = int(i);
		Parser p(res->cl.body_sql, sym);
		res->cl.ast = p.expr();
		return res;
	}
	bool Equals(const FunctionData &) const override { return false; }
};

static unique_ptr<FunctionData> MyGradEvalBindFunc(ClientContext &, ScalarFunction &func,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 2) {
		throw BinderException("mygrad_eval expects (LIST<DOUBLE>, LAMBDA)");
	}
	auto &t0 = arguments[0]->return_type;
	if (t0.id() != LogicalTypeId::LIST || ListType::GetChildType(t0).id() != LogicalTypeId::DOUBLE) {
		throw BinderException("mygrad_eval: first argument must be LIST<DOUBLE>");
	}
	if (arguments[1]->expression_class != ExpressionClass::LAMBDA) {
		throw BinderException("mygrad_eval: second argument must be a LAMBDA");
	}

	auto &lam = arguments[1]->Cast<LambdaExpression>();
	auto bind = make_uniq<MyGradEvalBind>();
	bind->cl = CompileDuckDBLambda(lam);

	func.return_type = LogicalType::DOUBLE;
	return std::move(bind);
}

static void MyGradEvalExecFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = state.bind_data->Cast<MyGradEvalBind>();
	const idx_t n = args.size();

	auto &list_vec = args.data[0];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto *out = FlatVector::GetData<double>(result);

	UnifiedVectorFormat lvf;
	list_vec.ToUnifiedFormat(n, lvf);

	for (idx_t row = 0; row < n; row++) {
		auto ridx = lvf.sel.get_index(row);
		if (!lvf.validity.RowIsValid(ridx)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto entry = ListVector::GetList(list_vec, ridx);
		auto &child = ListVector::GetEntry(list_vec);
		auto *cptr = FlatVector::GetData<double>(child);

		if (entry.length != bind.cl.params.size()) {
			throw BinderException("mygrad_eval: values list length (%llu) != lambda arity (%llu)",
				(unsigned long long)entry.length, (unsigned long long)bind.cl.params.size());
		}
		std::vector<double> x(entry.length);
		for (idx_t i=0;i<entry.length;i++) x[i] = cptr[entry.offset + i];

		auto dual = bind.cl.ast->eval(x, x.size());
		out[row] = dual.v;
	}
	result.SetCardinality(n);
}

// ---------- mygrad_grad ----------
struct MyGradGradBind : public FunctionData {
	CompiledLambda cl;
	unique_ptr<FunctionData> Copy() const override {
		auto res = make_uniq<MyGradGradBind>();
		res->cl.params = cl.params;
		res->cl.body_sql = cl.body_sql;
		ParamMap sym;
		for (idx_t i=0;i<res->cl.params.size();i++) sym[res->cl.params[i]] = int(i);
		Parser p(res->cl.body_sql, sym);
		res->cl.ast = p.expr();
		return res;
	}
	bool Equals(const FunctionData &) const override { return false; }
};

static unique_ptr<FunctionData> MyGradGradBindFunc(ClientContext &, ScalarFunction &func,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 2) {
		throw BinderException("mygrad_grad expects (LIST<DOUBLE>, LAMBDA)");
	}
	auto &t0 = arguments[0]->return_type;
	if (t0.id() != LogicalTypeId::LIST || ListType::GetChildType(t0).id() != LogicalTypeId::DOUBLE) {
		throw BinderException("mygrad_grad: first argument must be LIST<DOUBLE>");
	}
	if (arguments[1]->expression_class != ExpressionClass::LAMBDA) {
		throw BinderException("mygrad_grad: second argument must be a LAMBDA");
	}

	auto &lam = arguments[1]->Cast<LambdaExpression>();
	auto bind = make_uniq<MyGradGradBind>();
	bind->cl = CompileDuckDBLambda(lam);

	func.return_type = LogicalType::STRUCT({
		{"result", LogicalType::DOUBLE},
		{"grad",   LogicalType::LIST(LogicalType::DOUBLE)}
	});
	return std::move(bind);
}

static void MyGradGradExecFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind = state.bind_data->Cast<MyGradGradBind>();
	const idx_t n = args.size();

	// Prepare STRUCT children
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	D_ASSERT(children.size() == 2);
	auto &res_vec  = *children[0];
	auto &grad_vec = *children[1];

	res_vec.SetVectorType(VectorType::FLAT_VECTOR);
	grad_vec.SetVectorType(VectorType::FLAT_VECTOR);

	auto *res_out = FlatVector::GetData<double>(res_vec);

	auto &list_vec = args.data[0];
	UnifiedVectorFormat lvf;
	list_vec.ToUnifiedFormat(n, lvf);

	// Child of grad list
	auto &grad_child = ListVector::GetEntry(grad_vec);
	idx_t running_offset = 0;

	for (idx_t row = 0; row < n; row++) {
		auto ridx = lvf.sel.get_index(row);
		if (!lvf.validity.RowIsValid(ridx)) {
			StructVector::SetNull(result, row, true);
			continue;
		}

		auto entry = ListVector::GetList(list_vec, ridx);
		auto &child = ListVector::GetEntry(list_vec);
		auto *cptr = FlatVector::GetData<double>(child);

		if (entry.length != bind.cl.params.size()) {
			throw BinderException("mygrad_grad: values list length (%llu) != lambda arity (%llu)",
				(unsigned long long)entry.length, (unsigned long long)bind.cl.params.size());
		}

		std::vector<double> x(entry.length);
		for (idx_t i=0;i<entry.length;i++) x[i] = cptr[entry.offset + i];

		auto dual = bind.cl.ast->eval(x, x.size());
		res_out[row] = dual.v;

		// write grad list slice
		ListVector::Reserve(grad_vec, running_offset + dual.g.size());
		ListVector::SetList(grad_vec, row, {running_offset, (idx_t)dual.g.size()});
		auto *gptr = FlatVector::GetData<double>(grad_child);
		for (idx_t i=0;i<dual.g.size();i++) gptr[running_offset + i] = dual.g[i];
		running_offset += dual.g.size();
	}

	ListVector::SetListSize(grad_vec, running_offset);
	result.SetCardinality(n);
}

// Registration
void RegisterMyGradLambda(BuiltinFunctions &set) {
	// mygrad_eval
	{
		ScalarFunction fun(
			"mygrad_eval",
			{LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LAMBDA},
			LogicalType::DOUBLE,
			MyGradEvalExecFunc,
			MyGradEvalBindFunc
		);
		set.AddFunction(fun);
	}
	// mygrad_grad
	{
		LogicalType ret = LogicalType::STRUCT({
			{"result", LogicalType::DOUBLE},
			{"grad",   LogicalType::LIST(LogicalType::DOUBLE)}
		});
		ScalarFunction fun(
			"mygrad_grad",
			{LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LAMBDA},
			ret,
			MyGradGradExecFunc,
			MyGradGradBindFunc
		);
		set.AddFunction(fun);
	}
}

} // namespace duckdb
