#!/bin/bash
set -e
DUCKDB_BIN="./build/release/duckdb"
SIZES=(1000 5000 10000 20000 50000 100000 200000 500000 1000000)
RUNS=5
OUT_CSV="loss_function_scaling.csv"

if [ ! -x "$DUCKDB_BIN" ]; then
    echo "ERROR: $DUCKDB_BIN not found or not executable."
    exit 1
fi

echo "tuples,method,run,seconds" > "$OUT_CSV"

run_one() {
    local n="$1"
    local sql_file="$2"
    local method="$3"
    local tmpsql="/tmp/bench_tmp_${method}_${n}.sql"
    sed "s/__N__/${n}/g" "$sql_file" > "$tmpsql"
    for run in $(seq 1 $RUNS); do
        local start=$(date +%s.%N)
        "$DUCKDB_BIN" < "$tmpsql" > /tmp/bench_out.log 2>&1
        local end=$(date +%s.%N)
        local elapsed=$(awk "BEGIN {printf \"%.4f\", $end - $start}")
        echo "${n},${method},${run},${elapsed}" >> "$OUT_CSV"
        echo "  [$method] N=$n run=$run: ${elapsed}s"
    done
}

for n in "${SIZES[@]}"; do
    echo "=== N=$n ==="
    run_one "$n" bench_baseline_logistic.sql.in baseline_logistic
    run_one "$n" bench_fwd_logistic.sql.in      fwd_logistic
    run_one "$n" bench_rev_logistic.sql.in      rev_logistic
    run_one "$n" bench_baseline_poisson.sql.in  baseline_poisson
    run_one "$n" bench_fwd_poisson.sql.in       fwd_poisson
    run_one "$n" bench_rev_poisson.sql.in       rev_poisson
done

echo "DONE - raw data in $OUT_CSV"
