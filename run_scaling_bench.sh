#!/usr/bin/env bash
set -euo pipefail

DUCKDB_BIN="./build_debug/duckdb"
SQL_IN="bench_lr_fwd.sql.in"

SIZES=(1000 5000 10000 20000 50000)
REPEAT=3

OUTCSV="forward_ad_scaling.csv"

echo "tuples,run,seconds" > "${OUTCSV}"

for n in "${SIZES[@]}"; do
  for r in $(seq 1 "${REPEAT}"); do
    echo "Running N=${n}, run=${r}"

    SQL_TMP="$(mktemp --suffix=.sql)"
    TIME_TMP="$(mktemp --suffix=.time)"

    sed "s/__N__/${n}/g" "${SQL_IN}" > "${SQL_TMP}"

    /usr/bin/time -f "%e" -o "${TIME_TMP}" \
      "${DUCKDB_BIN}" < "${SQL_TMP}" > /dev/null 2> bench.err

    echo "${n},${r},$(cat "${TIME_TMP}")" >> "${OUTCSV}"

    rm -f "${SQL_TMP}" "${TIME_TMP}"
  done
done

echo "DONE"
echo "CSV written to ${OUTCSV}"
