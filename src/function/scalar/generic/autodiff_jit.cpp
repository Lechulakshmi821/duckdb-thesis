#ifdef DUCKDB_HAVE_LLVM
#include <vector>
#include <string>
#include <cstdint>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/Support/TargetSelect.h>

enum class FwdOpKind_jit : uint8_t { INPUT, CONST, ADD, SUB, MUL, DIV, NEG, POW };
struct FwdOp_jit {
    FwdOpKind_jit op;
    int32_t a = -1, b = -1, input_slot = -1;
    double cval = 0.0;
};

using JitFuncType = void(*)(const double*, double*);

JitFuncType CompileJITImpl(const std::vector<FwdOp_jit> &prog, int32_t root, uint64_t N) {
    using namespace llvm;
    static bool init = false;
    if (!init) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        InitializeNativeTargetAsmParser();
        init = true;
    }
    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("fwd_jit", *ctx);
    IRBuilder<> B(*ctx);
    Type *dbl = Type::getDoubleTy(*ctx);
    Type *dblptr = PointerType::get(dbl, 0);
    FunctionType *ft = FunctionType::get(Type::getVoidTy(*ctx), {dblptr, dblptr}, false);
    Function *fn = Function::Create(ft, Function::ExternalLinkage, "fwd_jit", mod.get());
    auto arg_it = fn->arg_begin();
    Value *inp = &*arg_it++;
    Value *out = &*arg_it++;
    BasicBlock *bb = BasicBlock::Create(*ctx, "entry", fn);
    B.SetInsertPoint(bb);

    const uint64_t sz = (uint64_t)root + 1;
    std::vector<Value*> val(sz);
    std::vector<std::vector<Value*>> grad(sz, std::vector<Value*>(N));
    auto C = [&](double v) -> Value* { return ConstantFP::get(dbl, v); };

    for (uint64_t i = 0; i < sz; i++) {
        const auto &op = prog[i];
        switch (op.op) {
        case FwdOpKind_jit::INPUT: {
            Value *ptr = B.CreateGEP(dbl, inp, B.getInt64(op.input_slot));
            val[i] = B.CreateLoad(dbl, ptr);
            for (uint64_t j = 0; j < N; j++)
                grad[i][j] = (j==(uint64_t)op.input_slot) ? C(1.0) : C(0.0);
            break;
        }
        case FwdOpKind_jit::CONST:
            val[i] = C(op.cval);
            for (uint64_t j = 0; j < N; j++) grad[i][j] = C(0.0);
            break;
        case FwdOpKind_jit::NEG:
            val[i] = B.CreateFNeg(val[op.a]);
            for (uint64_t j = 0; j < N; j++) grad[i][j] = B.CreateFNeg(grad[op.a][j]);
            break;
        case FwdOpKind_jit::ADD:
            val[i] = B.CreateFAdd(val[op.a], val[op.b]);
            for (uint64_t j = 0; j < N; j++) grad[i][j] = B.CreateFAdd(grad[op.a][j], grad[op.b][j]);
            break;
        case FwdOpKind_jit::SUB:
            val[i] = B.CreateFSub(val[op.a], val[op.b]);
            for (uint64_t j = 0; j < N; j++) grad[i][j] = B.CreateFSub(grad[op.a][j], grad[op.b][j]);
            break;
        case FwdOpKind_jit::MUL: {
            Value *lv=val[op.a], *rv=val[op.b];
            val[i] = B.CreateFMul(lv, rv);
            for (uint64_t j = 0; j < N; j++)
                grad[i][j] = B.CreateFAdd(B.CreateFMul(grad[op.a][j],rv), B.CreateFMul(grad[op.b][j],lv));
            break;
        }
        case FwdOpKind_jit::DIV: {
            Value *lv=val[op.a], *rv=val[op.b];
            Value *inv = B.CreateFDiv(C(1.0), rv);
            val[i] = B.CreateFMul(lv, inv);
            for (uint64_t j = 0; j < N; j++) {
                Value *num = B.CreateFSub(B.CreateFMul(grad[op.a][j],rv), B.CreateFMul(grad[op.b][j],lv));
                grad[i][j] = B.CreateFMul(num, B.CreateFMul(inv,inv));
            }
            break;
        }
        default:
            val[i] = C(0.0);
            for (uint64_t j = 0; j < N; j++) grad[i][j] = C(0.0);
            break;
        }
    }
    for (uint64_t j = 0; j < N; j++) {
        Value *ptr = B.CreateGEP(dbl, out, B.getInt64(j));
        B.CreateStore(grad[root][j], ptr);
    }
    B.CreateRetVoid();

    std::string err;
    ExecutionEngine *ee = EngineBuilder(std::move(mod))
        .setErrorStr(&err).setEngineKind(EngineKind::JIT).create();
    if (!ee) return nullptr;
    ee->finalizeObject();
    return (JitFuncType)(uintptr_t)ee->getFunctionAddress("fwd_jit");
}
#endif

#ifdef DUCKDB_HAVE_LLVM
// Bridge function callable from duckdb namespace
JitFuncType CompileJITBridge(const void* prog_ptr, int32_t root, uint64_t N) {
    const auto* prog = static_cast<const std::vector<FwdOp_jit>*>(prog_ptr);
    fprintf(stderr, "[JIT bridge] prog size=%zu root=%d N=%lu\n", prog->size(), root, N);
    auto r = CompileJITImpl(*prog, root, N);
    fprintf(stderr, "[JIT bridge] result=%p\n", (void*)r);
    return r;
}
#endif
