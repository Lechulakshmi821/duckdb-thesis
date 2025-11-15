import matplotlib.pyplot as plt

# Actual timings from DuckDB benchmark
baseline_time = 0.002     # Baseline SQL SUM(...)
mygrad_time   = 0.168     # mygrad_lambda(...)

methods = ["Baseline SQL", "mygrad_lambda"]
times = [baseline_time, mygrad_time]

plt.figure(figsize=(6, 4))
plt.bar(methods, times)
plt.ylabel("Runtime (seconds)")
plt.title("Performance Comparison: Baseline vs mygrad_lambda (100k rows)")


for i, t in enumerate(times):
    plt.text(i, t, f"{t:.3f}s", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("bench_compare.png", dpi=200)
print("Saved plot to bench_compare.png")
