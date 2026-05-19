#!/usr/bin/env bash
set -euo pipefail

DUCKDB_BIN="./build/release/duckdb"
SIZES=(1000 5000 10000 20000 50000)
REPEAT=5
OUTCSV="ad_comparison_scaling.csv"

if [[ ! -x "${DUCKDB_BIN}" ]]; then
  echo "ERROR: ${DUCKDB_BIN} not found or not executable." >&2
  exit 1
fi

echo "tuples,method,run,seconds" > "${OUTCSV}"

run_one() {
  local n=$1 method=$2 sql_file=$3 run=$4
  local sql_tmp time_tmp err_tmp
  sql_tmp="$(mktemp --suffix=.sql)"
  time_tmp="$(mktemp --suffix=.time)"
  err_tmp="$(mktemp --suffix=.err)"

  echo "  ${method}  N=${n}  run=${run}"
  sed "s/__N__/${n}/g" "${sql_file}" > "${sql_tmp}"

  if ! /usr/bin/time -f "%e" -o "${time_tmp}" \
        "${DUCKDB_BIN}" < "${sql_tmp}" > /dev/null 2> "${err_tmp}"; then
    echo "ERROR: ${method} failed at N=${n}, run=${run}" >&2
    echo "----- DuckDB stderr -----" >&2
    cat "${err_tmp}" >&2
    rm -f "${sql_tmp}" "${time_tmp}" "${err_tmp}"
    exit 1
  fi

  echo "${n},${method},${run},$(cat "${time_tmp}")" >> "${OUTCSV}"
  rm -f "${sql_tmp}" "${time_tmp}" "${err_tmp}"
}

for n in "${SIZES[@]}"; do
  echo "Warmup N=${n}"
  sed "s/__N__/${n}/g" bench_baseline.sql.in \
    | "${DUCKDB_BIN}" > /dev/null 2>&1 || true
done

for n in "${SIZES[@]}"; do
  for r in $(seq 1 "${REPEAT}"); do
    run_one "${n}" "baseline"   "bench_baseline.sql.in"          "${r}"
    run_one "${n}" "forward_ad" "bench_fwd_single_call.sql.in"   "${r}"
    run_one "${n}" "reverse_ad" "bench_rev_single_call.sql.in"   "${r}"
  done
done

echo "DONE"
echo "Raw per-run CSV written to ${OUTCSV}"
