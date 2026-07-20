#!/usr/bin/env python3
import csv
import statistics
from collections import defaultdict

IN_CSV = "stored_lambda_gd_scaling.csv"
OUT_CSV = "stored_lambda_gd_summary.csv"


def main():
    rows_by_method = defaultdict(list)

    with open(IN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            try:
                seconds = float(row["seconds"])
            except (ValueError, TypeError):
                continue
            rows_by_method[method].append(seconds)

    if not rows_by_method:
        print(f"No valid rows found in {IN_CSV}. Did any run fail?")
        return

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_runs", "median_seconds", "min_seconds", "max_seconds"])

        for method, values in rows_by_method.items():
            median = statistics.median(values)
            writer.writerow([method, len(values), f"{median:.4f}", f"{min(values):.4f}", f"{max(values):.4f}"])
            print(f"{method}: n={len(values)} median={median:.4f}s min={min(values):.4f}s max={max(values):.4f}s")

    print(f"\nWrote summary to {OUT_CSV}")


if __name__ == "__main__":
    main()
