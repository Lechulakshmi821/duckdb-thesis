#!/usr/bin/env python3
import csv
import statistics
from collections import defaultdict

IN_CSV = "stored_lambda_scaling.csv"
OUT_CSV = "stored_lambda_scaling_summary.csv"


def main():
    rows = defaultdict(list)

    with open(IN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tuples = int(row["tuples"])
                seconds = float(row["seconds"])
            except (ValueError, TypeError, KeyError):
                continue
            method = row["method"]
            rows[(tuples, method)].append(seconds)

    if not rows:
        print(f"No valid rows found in {IN_CSV}. Did any run fail?")
        return

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tuples", "method", "n_runs", "median_seconds", "min_seconds", "max_seconds"])

        for (tuples, method), values in sorted(rows.items()):
            median = statistics.median(values)
            writer.writerow([tuples, method, len(values), f"{median:.4f}", f"{min(values):.4f}", f"{max(values):.4f}"])
            print(f"tuples={tuples} {method}: n={len(values)} median={median:.4f}s min={min(values):.4f}s max={max(values):.4f}s")

    print(f"\nWrote summary to {OUT_CSV}")


if __name__ == "__main__":
    main()
