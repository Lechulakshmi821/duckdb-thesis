#!/bin/bash
set -e

DUCKDB_BIN="./build/release/duckdb"
SQL_FILE="bench_stored_lambda.sql.in"
OUT_CSV="stored_lambda_scaling.csv"
RUNS=5

if [ ! -x "$DUCKDB_BIN" ]; then
    echo "ERROR: $DUCKDB_BIN not found or not executable."
    exit 1
fi

echo "method,run,seconds" > "$OUT_CSV"

for run in $(seq 1 $RUNS); do
    LOG="/tmp/stored_lambda_run_${run}.log"
    "$DUCKDB_BIN" < "$SQL_FILE" > "$LOG" 2>&1 || true

    TIMES=($(grep "Run Time" "$LOG" | tail -n 4 | awk '{print $(NF-4)}'))

    if grep -q "Error:" "$LOG"; then
        echo "Run $run: SQL ERROR -- see $LOG"
        grep "Error:" "$LOG"
        continue
    fi

    echo "inline_static,$run,${TIMES[0]}" >> "$OUT_CSV"
    echo "stored_constant,$run,${TIMES[1]}" >> "$OUT_CSV"
    echo "stored_dynamic,$run,${TIMES[2]}" >> "$OUT_CSV"

    MISMATCH=$(grep -A3 "mismatched_rows" "$LOG" | tail -n1 | grep -oE '[0-9]+' | head -n1)

    echo "Run $run: inline=${TIMES[0]}s constant=${TIMES[1]}s dynamic=${TIMES[2]}s | mismatched_rows=${MISMATCH:-?}"
    if [ "${MISMATCH:-0}" != "0" ]; then
        echo "  !! WARNING: dynamic path disagreed with / NaN'd on ${MISMATCH} rows this run."
    fi
done

echo "Done. Raw timings in $OUT_CSV"
