#pragma once
#include "duckdb.hpp"
#include <vector>
#include <string>

namespace duckdb {

enum class StoredOpKind : uint8_t { INPUT, CONST, NEG, ADD, SUB, MUL, DIV, POW };
struct StoredCompiledOp {
	StoredOpKind op;
	int32_t a = -1, b = -1;
	double cval = 0.0;
	int32_t input_slot = -1;
};

struct ParsedLambdaTape {
	std::vector<StoredCompiledOp> prog;
	int32_t root = -1;
	idx_t n_params = 0;
};

ParsedLambdaTape CompileStoredLambda(const std::string &lambda_text);

} // namespace duckdb
