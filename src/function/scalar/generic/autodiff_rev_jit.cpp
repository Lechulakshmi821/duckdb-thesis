#ifdef DUCKDB_HAVE_LLVM
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Intrinsics.h>

// Mirror of CompiledOp from autodiff_reverse.cpp
enum class RevOpKind_jit : uint8_t { INPUT, CONST, ADD, SUB, MUL, DIV, NEG, POW };
struct RevOp_jit {
    RevOpKind_jit op;
    int32_t a = -1, b = -1;
    int32_t input_slot = -1;
    double cval = 0.0;
};

using RevJitFuncType = void(*)(const double*, double*);

RevJitFuncType CompileRevJITImpl(
    const std::vector<RevOp_jit> &prog,
    int32_t root,
    uint64_t N,
    const std::vector<int32_t> &input_node) {

    using namespace llvm;
    static bool init = false;
    if (!init) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        InitializeNativeTargetAsmParser();
        init = true;
    }

    auto ctx = std::make_unique<LLVMContext>();
    auto mod = std::make_unique<Module>("rev_tape_jit", *ctx);
    IRBuilder<> B(*ctx);
    Type *dbl = Type::getDoubleTy(*ctx);
    Type *dblptr = PointerType::get(dbl, 0);

    // void f(const double* inputs, double* gradients)
    FunctionType *ft = FunctionType::get(Type::getVoidTy(*ctx), {dblptr, dblptr}, false);
    Function *fn = Function::Create(ft, Function::ExternalLinkage, "rev_tape_jit", mod.get());
    auto arg_it = fn->arg_begin();
    Value *inp = &*arg_it++;
    Value *out = &*arg_it++;

    BasicBlock *bb = BasicBlock::Create(*ctx, "entry", fn);
    B.SetInsertPoint(bb);

    const uint64_t sz = prog.size();
    auto C = [&](double v) -> Value* { return ConstantFP::get(dbl, v); };
    auto Zero = C(0.0);

    // === FORWARD PASS: generate val[] as LLVM values ===
    std::vector<Value*> val(sz, nullptr);
    for (uint64_t i = 0; i < sz; i++) {
        const auto &op = prog[i];
        switch (op.op) {
        case RevOpKind_jit::INPUT: {
            Value *ptr = B.CreateGEP(dbl, inp, B.getInt64(op.input_slot));
            val[i] = B.CreateLoad(dbl, ptr);
            break;
        }
        case RevOpKind_jit::CONST:
            val[i] = C(op.cval); break;
        case RevOpKind_jit::NEG:
            val[i] = B.CreateFNeg(val[(uint64_t)op.a]); break;
        case RevOpKind_jit::ADD:
            val[i] = B.CreateFAdd(val[(uint64_t)op.a], val[(uint64_t)op.b]); break;
        case RevOpKind_jit::SUB:
            val[i] = B.CreateFSub(val[(uint64_t)op.a], val[(uint64_t)op.b]); break;
        case RevOpKind_jit::MUL:
            val[i] = B.CreateFMul(val[(uint64_t)op.a], val[(uint64_t)op.b]); break;
        case RevOpKind_jit::DIV:
            val[i] = B.CreateFDiv(val[(uint64_t)op.a], val[(uint64_t)op.b]); break;
        case RevOpKind_jit::POW: {
            Function *pow_fn = Intrinsic::getDeclaration(mod.get(), Intrinsic::pow, {dbl});
            val[i] = B.CreateCall(pow_fn, {val[(uint64_t)op.a], val[(uint64_t)op.b]});
            break;
        }
        default: val[i] = Zero; break;
        }
    }

    // === BACKWARD PASS: use alloca for mutable adjoints ===
    std::vector<Value*> adj(sz);
    for (uint64_t i = 0; i < sz; i++) {
        adj[i] = B.CreateAlloca(dbl, nullptr);
        B.CreateStore(Zero, adj[i]);
    }
    if (root >= 0) B.CreateStore(C(1.0), adj[(uint64_t)root]);

    // Backward pass — unrolled, each op accumulates into adj
    for (int64_t ii = (int64_t)sz - 1; ii >= 0; ii--) {
        uint64_t i = (uint64_t)ii;
        const auto &op = prog[i];
        Value *a_i = B.CreateLoad(dbl, adj[i]);

        switch (op.op) {
        case RevOpKind_jit::INPUT:
        case RevOpKind_jit::CONST:
            break;
        case RevOpKind_jit::NEG: {
            uint64_t pa = (uint64_t)op.a;
            Value *cur = B.CreateLoad(dbl, adj[pa]);
            B.CreateStore(B.CreateFAdd(cur, B.CreateFNeg(a_i)), adj[pa]);
            break;
        }
        case RevOpKind_jit::ADD: {
            uint64_t pa=(uint64_t)op.a, pb=(uint64_t)op.b;
            if (pa == pb) {
                Value *ca = B.CreateLoad(dbl, adj[pa]);
                B.CreateStore(B.CreateFAdd(ca, B.CreateFMul(C(2.0), a_i)), adj[pa]);
            } else {
                Value *ca = B.CreateLoad(dbl, adj[pa]);
                Value *cb = B.CreateLoad(dbl, adj[pb]);
                B.CreateStore(B.CreateFAdd(ca, a_i), adj[pa]);
                B.CreateStore(B.CreateFAdd(cb, a_i), adj[pb]);
            }
            break;
        }
        case RevOpKind_jit::SUB: {
            uint64_t pa=(uint64_t)op.a, pb=(uint64_t)op.b;
            Value *ca = B.CreateLoad(dbl, adj[pa]);
            Value *cb = B.CreateLoad(dbl, adj[pb]);
            B.CreateStore(B.CreateFAdd(ca, a_i), adj[pa]);
            B.CreateStore(B.CreateFSub(cb, a_i), adj[pb]);
            break;
        }
        case RevOpKind_jit::MUL: {
            uint64_t pa=(uint64_t)op.a, pb=(uint64_t)op.b;
            Value *lv=val[pa], *rv=val[pb];
            if (pa == pb) {
                // Self-referential: adj[pa] += a_i*(lv+rv)
                Value *ca = B.CreateLoad(dbl, adj[pa]);
                B.CreateStore(B.CreateFAdd(ca, B.CreateFMul(a_i, B.CreateFAdd(lv,rv))), adj[pa]);
            } else {
                Value *ca = B.CreateLoad(dbl, adj[pa]);
                Value *cb = B.CreateLoad(dbl, adj[pb]);
                B.CreateStore(B.CreateFAdd(ca, B.CreateFMul(a_i,rv)), adj[pa]);
                B.CreateStore(B.CreateFAdd(cb, B.CreateFMul(a_i,lv)), adj[pb]);
            }
            break;
        }
        case RevOpKind_jit::DIV: {
            uint64_t pa=(uint64_t)op.a, pb=(uint64_t)op.b;
            Value *lv=val[pa], *rv=val[pb];
            Value *inv = B.CreateFDiv(C(1.0), rv);
            Value *ca = B.CreateLoad(dbl, adj[pa]);
            Value *cb = B.CreateLoad(dbl, adj[pb]);
            B.CreateStore(B.CreateFAdd(ca, B.CreateFMul(a_i,inv)), adj[pa]);
            Value *neg_lv_inv2 = B.CreateFMul(B.CreateFNeg(lv), B.CreateFMul(inv,inv));
            B.CreateStore(B.CreateFAdd(cb, B.CreateFMul(a_i,neg_lv_inv2)), adj[pb]);
            break;
        }
        case RevOpKind_jit::POW: {
            uint64_t pa=(uint64_t)op.a;
            Value *lv=val[pa], *rv=val[(uint64_t)op.b];
            Function *pow_fn = Intrinsic::getDeclaration(mod.get(), Intrinsic::pow, {dbl});
            Value *ca = B.CreateLoad(dbl, adj[pa]);
            Value *upow = B.CreateCall(pow_fn, {lv, B.CreateFSub(rv, C(1.0))});
            Value *da = B.CreateFMul(a_i, B.CreateFMul(rv, upow));
            B.CreateStore(B.CreateFAdd(ca, da), adj[pa]);
            break;
        }
        default: break;
        }
    }

    // Output gradients
    for (uint64_t j = 0; j < N; j++) {
        int32_t node = input_node[j];
        Value *grad = (node >= 0) ? B.CreateLoad(dbl, adj[(uint64_t)node]) : Zero;
        Value *ptr = B.CreateGEP(dbl, out, B.getInt64(j));
        B.CreateStore(grad, ptr);
    }
    B.CreateRetVoid();

    std::string err;
    ExecutionEngine *ee = EngineBuilder(std::move(mod))
        .setErrorStr(&err).setEngineKind(EngineKind::JIT).create();
    if (!ee) { fprintf(stderr, "[RevJIT] failed: %s\n", err.c_str()); return nullptr; }
    ee->finalizeObject();
    auto fp = (RevJitFuncType)(uintptr_t)ee->getFunctionAddress("rev_tape_jit");
    fprintf(stderr, "[RevJIT] %s\n", fp ? "tape JIT compiled OK" : "failed");
    return fp;
}

// Bridge callable from duckdb namespace
RevJitFuncType CompileRevJITBridge(
    const void* prog_ptr,
    const void* root_ptr,
    const void* input_node_ptr,
    uint64_t N) {
    const auto* prog = static_cast<const std::vector<RevOp_jit>*>(prog_ptr);
    int32_t root = *static_cast<const int32_t*>(root_ptr);
    const auto* input_node = static_cast<const std::vector<int32_t>*>(input_node_ptr);
    return CompileRevJITImpl(*prog, root, N, *input_node);
}
#endif
