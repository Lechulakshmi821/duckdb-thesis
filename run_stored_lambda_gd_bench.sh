#!/bin/bash
set -e

DUCKDB_BIN="./build/release/duckdb"
OUT_CSV="stored_lambda_gd_scaling.csv"
RUNS=5

if [ ! -x "$DUCKDB_BIN" ]; then
    echo "ERROR: $DUCKDB_BIN not found or not executable."
    exit 1
fi

echo "method,run,seconds" > "$OUT_CSV"

run_one() {
    local sql_file="$1"
    local method="$2"

    for run in $(seq 1 $RUNS); do
        local log="/tmp/${method}_run_${run}.log"
        "$DUCKDB_BIN" < "$sql_file" > "$log" 2>&1 || true

        if grep -q "Error:" "$log"; then
            echo "Run $run ($method): SQL ERROR -- see $log"
            grep "Error:" "$log"
            continue
        fi

        seconds=$(grep "Run Time" "$log" | tail -n1 | awk '{print $(NF-4)}')
        echo "${method},${run},${seconds}" >> "$OUT_CSV"
        echo "Run $run ($method): ${seconds}s"
    done
}

run_one "bench_stored_constant.sql.in" "stored_constant"
run_one "bench_stored_dynamic.sql.in"  "stored_dynamic"

echo ""
echo "Done. Raw results in $OUT_CSV"
echo "Run: python3 aggregate_stored_lambda.py"
