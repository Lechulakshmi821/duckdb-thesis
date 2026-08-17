import csv
from collections import defaultdict
import statistics

rows = defaultdict(list)
with open('loss_function_scaling.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row['tuples']), row['method'])
        rows[key].append(float(row['seconds']))

methods = ['baseline_logistic', 'fwd_logistic', 'rev_logistic',
           'baseline_poisson', 'fwd_poisson', 'rev_poisson']
sizes = sorted(set(k[0] for k in rows.keys()))

print(f"{'tuples':>9} | " + " | ".join(f"{m:>18}" for m in methods))
print("-" * 140)
with open('loss_function_summary.csv', 'w') as out:
    out.write("tuples," + ",".join(f"{m}_median" for m in methods) + "\n")
    for n in sizes:
        medians = []
        for m in methods:
            vals = rows.get((n, m), [])
            med = statistics.median(vals) if vals else float('nan')
            medians.append(med)
        print(f"{n:>9} | " + " | ".join(f"{v:>18.4f}" for v in medians))
        out.write(f"{n}," + ",".join(f"{v:.4f}" for v in medians) + "\n")

print("\nSummary written to loss_function_summary.csv")
