#!/bin/bash
set -e

DUCKDB_BIN="./build/release/duckdb"
SIZES=(1000 5000 10000 20000 50000 100000 200000 500000 1000000)
RUNS=5
OUT_CSV="stored_lambda_scaling.csv"

if [ ! -x "$DUCKDB_BIN" ]; then
    echo "ERROR: $DUCKDB_BIN not found or not executable."
    exit 1
fi

echo "tuples,method,run,seconds" > "$OUT_CSV"

run_one() {
    local n="$1"
    local sql_file="$2"
    local method="$3"

    local sql_tmp="/tmp/${method}_${n}.sql"
    sed "s/__N__/${n}/g" "${sql_file}" > "${sql_tmp}"

    for run in $(seq 1 $RUNS); do
        local log="/tmp/${method}_${n}_run_${run}.log"
        "$DUCKDB_BIN" < "$sql_tmp" > "$log" 2>&1 || true

        if grep -q "Error:" "$log"; then
            echo "n=$n run=$run ($method): SQL ERROR -- see $log"
            grep "Error:" "$log"
            continue
        fi

        local seconds=$(grep "Run Time" "$log" | tail -n1 | awk '{print $(NF-4)}')
        echo "${n},${method},${run},${seconds}" >> "$OUT_CSV"
        echo "n=$n run=$run ($method): ${seconds}s"
    done
}

for n in "${SIZES[@]}"; do
    run_one "$n" "bench_stored_constant.sql.in" "stored_constant_rev"
done
for n in "${SIZES[@]}"; do
    run_one "$n" "bench_stored_dynamic.sql.in" "stored_dynamic_rev"
done

for n in "${SIZES[@]}"; do
    run_one "$n" "bench_stored_constant_fwd.sql.in" "stored_constant_fwd"
done
for n in "${SIZES[@]}"; do
    run_one "$n" "bench_stored_dynamic_fwd.sql.in" "stored_dynamic_fwd"
done

echo ""
echo "Done. Raw results in $OUT_CSV"
echo "Run: python3 aggregate_stored_lambda_scaling.py"
