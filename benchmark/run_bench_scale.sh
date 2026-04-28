#!/usr/bin/env bash
set -euo pipefail

DUCKDB_BIN="./build_debug/duckdb"
SQL_IN="bench_lr.sql.in"

# Different dataset sizes
SIZES=(1000 5000 10000 20000 50000)

OUTCSV="lr_scaling.csv"

echo "n,seconds" > "${OUTCSV}"

for n in "${SIZES[@]}"; do
  echo "Running for N=${n}"

  SQL_TMP="$(mktemp)"
  TIME_TMP="$(mktemp)"

  sed "s/__N__/${n}/g" "${SQL_IN}" > "${SQL_TMP}"

  /usr/bin/time -f "%e" -o "${TIME_TMP}" \
    "${DUCKDB_BIN}" < "${SQL_TMP}" > /dev/null 2> bench.err

  SECONDS=$(cat "${TIME_TMP}")

  echo "${n},${SECONDS}" >> "${OUTCSV}"

  rm -f "${SQL_TMP}" "${TIME_TMP}"
done

echo "DONE: ${OUTCSV} created"
