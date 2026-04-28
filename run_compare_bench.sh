#!/usr/bin/env bash
set -euo pipefail

DUCKDB_BIN="./build_debug/duckdb"

SIZES=(1000 5000 10000 20000 50000)
REPEAT=3

OUTCSV="ad_comparison_scaling.csv"

echo "tuples,method,run,seconds" > "${OUTCSV}"

run_one() {
  local n=$1
  local method=$2
  local sql_file=$3
  local run=$4

  echo "Running ${method}, N=${n}, run=${run}"

  SQL_TMP="$(mktemp --suffix=.sql)"
  TIME_TMP="$(mktemp --suffix=.time)"

  sed "s/__N__/${n}/g" "${sql_file}" > "${SQL_TMP}"

  /usr/bin/time -f "%e" -o "${TIME_TMP}" \
    "${DUCKDB_BIN}" < "${SQL_TMP}" > /dev/null 2> bench.err

  echo "${n},${method},${run},$(cat "${TIME_TMP}")" >> "${OUTCSV}"

  rm -f "${SQL_TMP}" "${TIME_TMP}"
}

for n in "${SIZES[@]}"; do
  for r in $(seq 1 "${REPEAT}"); do
    run_one "${n}" "baseline" "bench_baseline.sql.in" "${r}"
    run_one "${n}" "forward_ad" "bench_fwd.sql.in" "${r}"
    run_one "${n}" "reverse_ad" "bench_rev.sql.in" "${r}"
  done
done

echo "DONE"
echo "CSV written to ${OUTCSV}"
