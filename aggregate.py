#!/usr/bin/env python3
import csv, statistics
from collections import defaultdict

IN = "ad_comparison_scaling.csv"
OUT = "ad_summary.csv"
METHODS = ["baseline", "forward_ad", "reverse_ad"]

data = defaultdict(lambda: defaultdict(list))
with open(IN) as f:
    for row in csv.DictReader(f):
        data[int(row["tuples"])][row["method"]].append(float(row["seconds"]))

sizes = sorted(data)
header = ["tuples"]
for m in METHODS:
    header += [f"{m}_median", f"{m}_min", f"{m}_max"]

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for n in sizes:
        line = [n]
        for m in METHODS:
            vals = data[n][m]
            if not vals:
                line += ["", "", ""]
                continue
            line += [round(statistics.median(vals), 4),
                     round(min(vals), 4), round(max(vals), 4)]
        w.writerow(line)

print(f"{'tuples':>8} | {'baseline':>10} | {'forward_ad':>10} | {'reverse_ad':>10}")
print("-" * 50)
for n in sizes:
    med = {m: statistics.median(data[n][m]) if data[n][m] else float('nan')
           for m in METHODS}
    print(f"{n:>8} | {med['baseline']:>10.4f} | "
          f"{med['forward_ad']:>10.4f} | {med['reverse_ad']:>10.4f}")
print()
for n in sizes:
    b = statistics.median(data[n]["baseline"])
    r = statistics.median(data[n]["reverse_ad"])
    fw = statistics.median(data[n]["forward_ad"])
    print(f"N={n:>6}: reverse/baseline = {r/b:5.2f}x   "
          f"forward/reverse = {fw/r:5.2f}x")
print(f"\nSummary written to {OUT}")
